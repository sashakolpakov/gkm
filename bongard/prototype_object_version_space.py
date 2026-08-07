"""Finite support-consistent Python predicates over stable object lineages.

Vision may nominate feature *identifiers*, but it cannot invent thresholds or
code.  This module deterministically enumerates a closed positive language,
evaluates every conjunction on one stable object lineage, and retains exactly
the candidates that are present on every positive support scene and certified
absent on every negative support scene.

The input records contain no support-side label.  Sides are supplied only to
``build_object_support_version_space``.  Missing or unresolved lineages never
become negative evidence.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
from itertools import combinations, product
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.prototype_object_profiles import (
    OBJECT_FEATURE_CATALOG,
    OBJECT_FEATURE_CATALOG_DIGEST,
    OBJECT_FEATURE_IDS,
    IntegerInterval,
    ObjectProfile,
    ObjectProfileAtom,
    ObjectProfileOperator,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


OBJECT_PREDICATE_GRID_SCHEMA = "gkm.bongard-object-predicate-grid.v1"
OBJECT_SCENE_FEATURE_VALUE_SCHEMA = "gkm.bongard-object-scene-feature-value.v1"
OBJECT_STABLE_LINEAGE_EVIDENCE_SCHEMA = (
    "gkm.bongard-object-stable-lineage-evidence.v1"
)
OBJECT_SCENE_EVIDENCE_SCHEMA = "gkm.bongard-object-scene-evidence.v1"
OBJECT_CANDIDATE_EVALUATION_SCHEMA = (
    "gkm.bongard-object-candidate-scene-evaluation.v1"
)
OBJECT_SUPPORT_DIAGNOSTIC_SCHEMA = "gkm.bongard-object-support-diagnostic.v1"
OBJECT_SUPPORT_GAP_SCHEMA = "gkm.bongard-object-support-gap.v1"
OBJECT_SUPPORT_VERSION_SPACE_SCHEMA = (
    "gkm.bongard-object-support-version-space.v1"
)
OBJECT_VERSION_SPACE_ALGORITHM_ID = (
    "bongard.object-support-version-space/positive-at-least-lineage-v1"
)

COUNT_THRESHOLDS = (1, 2, 3, 4)
PPM_THRESHOLDS = (250_000, 500_000, 750_000, 1_000_000)
MAX_CONJUNCTION_ATOMS = 2

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}\Z")
_SCENE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")
_FEATURE_INDEX = {feature_id: index for index, feature_id in enumerate(OBJECT_FEATURE_IDS)}
_FEATURE_SPEC = {item.feature_id: item for item in OBJECT_FEATURE_CATALOG}


class ObjectVersionSpaceError(ValueError):
    """A grid, evidence record, evaluation, or version space is malformed."""


class ObjectSupportGapKind(str, Enum):
    LANGUAGE_GAP = "language_gap"
    WITNESS_GAP = "witness_gap"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectVersionSpaceError(f"{label} fields differ from the closed schema")
    return value


def _code(value: object, label: str) -> str:
    if not isinstance(value, str) or _CODE.fullmatch(value) is None:
        raise ObjectVersionSpaceError(f"{label} must be a bounded code")
    return value


def _scene_id(value: object, label: str = "scene_id") -> str:
    if not isinstance(value, str) or _SCENE_ID.fullmatch(value) is None:
        raise ObjectVersionSpaceError(f"{label} must be a bounded panel identifier")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ObjectVersionSpaceError(f"{label} must be a lowercase sha256")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise ObjectVersionSpaceError(f"{label} must be null or nonempty stripped text")
    return value


def _canonical_feature_ids(values: Sequence[str], *, allow_empty: bool) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(item, str) or item not in _FEATURE_INDEX for item in result):
        raise ObjectVersionSpaceError("allowed feature is outside the frozen catalog")
    expected = tuple(sorted(set(result), key=_FEATURE_INDEX.__getitem__))
    if result != expected:
        raise ObjectVersionSpaceError(
            "allowed features must be unique and in frozen catalog order"
        )
    if not allow_empty and not result:
        raise ObjectVersionSpaceError("feature inventory must be nonempty")
    return result


def object_version_space_algorithm_digest() -> str:
    """Bind the closed language and evaluator to their exact loaded source."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-object-version-space-algorithm.v1",
            "algorithm_id": OBJECT_VERSION_SPACE_ALGORITHM_ID,
            "implementation_source_sha256": _LOADED_SOURCE_SHA256,
            "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
            "count_thresholds": list(COUNT_THRESHOLDS),
            "ppm_thresholds": list(PPM_THRESHOLDS),
            "operator": ObjectProfileOperator.AT_LEAST.value,
            "max_conjunction_atoms": MAX_CONJUNCTION_ATOMS,
            "same_lineage_conjunction": True,
            "candidate_order": "arity_then_catalog_features_then_thresholds",
            "positive_accept": Disposition.PRESENT.value,
            "negative_accept": Disposition.CERTIFIED_ABSENT.value,
            "missing_or_unresolved_is_absent": False,
            "negation": False,
            "polarity_flip": False,
            "disjunction": False,
            "arbitrary_code": False,
            **_authority_data(),
        }
    )


def _grid_content(value: "ObjectPredicateGrid") -> dict[str, object]:
    return {
        "schema": OBJECT_PREDICATE_GRID_SCHEMA,
        "algorithm_id": OBJECT_VERSION_SPACE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
        "allowed_feature_ids": list(value.allowed_feature_ids),
        "count_thresholds": list(COUNT_THRESHOLDS),
        "ppm_thresholds": list(PPM_THRESHOLDS),
        "operator": ObjectProfileOperator.AT_LEAST.value,
        "max_conjunction_atoms": MAX_CONJUNCTION_ATOMS,
        "no_nomination_means_language_gap": True,
        "complete_finite_inventory": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectPredicateGrid:
    """The exact finite language admitted for one task."""

    allowed_feature_ids: tuple[str, ...]
    algorithm_digest: str
    grid_digest: str

    def __post_init__(self) -> None:
        _canonical_feature_ids(self.allowed_feature_ids, allow_empty=True)
        if self.algorithm_digest != object_version_space_algorithm_digest():
            raise ObjectVersionSpaceError("grid algorithm binding differs")
        _digest(self.grid_digest, "grid_digest")
        if self.grid_digest != canonical_digest(_grid_content(self)):
            raise ObjectVersionSpaceError("grid digest differs from canonical grid")

    @classmethod
    def create(cls, allowed_feature_ids: Sequence[str]) -> "ObjectPredicateGrid":
        allowed = tuple(allowed_feature_ids)
        _canonical_feature_ids(allowed, allow_empty=True)
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "allowed_feature_ids", allowed)
        object.__setattr__(
            provisional, "algorithm_digest", object_version_space_algorithm_digest()
        )
        return cls(
            allowed,
            object_version_space_algorithm_digest(),
            canonical_digest(_grid_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_grid_content(self), "grid_digest": self.grid_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectPredicateGrid":
        expected = {
            "schema",
            "algorithm_id",
            "algorithm_digest",
            "feature_catalog_digest",
            "allowed_feature_ids",
            "count_thresholds",
            "ppm_thresholds",
            "operator",
            "max_conjunction_atoms",
            "no_nomination_means_language_gap",
            "complete_finite_inventory",
            *_authority_data(),
            "grid_digest",
        }
        raw = _fields(value, expected, "object predicate grid")
        if not isinstance(raw["allowed_feature_ids"], list):
            raise ObjectVersionSpaceError("allowed_feature_ids must be a JSON list")
        result = cls(
            tuple(raw["allowed_feature_ids"]),
            raw["algorithm_digest"],
            raw["grid_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectVersionSpaceError("grid is not canonical")
        return result


def _thresholds(feature_id: str) -> tuple[int, ...]:
    return PPM_THRESHOLDS if _FEATURE_SPEC[feature_id].unit == "ppm" else COUNT_THRESHOLDS


def _profile_id(atoms: Sequence[ObjectProfileAtom]) -> str:
    encoded = ".and.".join(f"{item.feature_id}:{item.target}" for item in atoms)
    return f"object-vs:{encoded}"


def enumerate_object_profile_candidates(
    grid: ObjectPredicateGrid,
) -> tuple[ObjectProfile, ...]:
    """Enumerate all one- and two-feature positive ``AT_LEAST`` profiles."""

    grid = ObjectPredicateGrid.from_data(grid.to_data())
    atoms_by_feature = tuple(
        tuple(
            ObjectProfileAtom(feature_id, ObjectProfileOperator.AT_LEAST, target)
            for target in _thresholds(feature_id)
        )
        for feature_id in grid.allowed_feature_ids
    )
    singles = tuple(
        ObjectProfile.create(_profile_id((atom,)), (atom,))
        for feature_atoms in atoms_by_feature
        for atom in feature_atoms
    )
    pairs = tuple(
        ObjectProfile.create(_profile_id((left, right)), (left, right))
        for left_atoms, right_atoms in combinations(atoms_by_feature, 2)
        for left, right in product(left_atoms, right_atoms)
    )
    return singles + pairs


@dataclass(frozen=True, slots=True)
class ObjectSceneFeatureValue:
    feature_id: str
    disposition: Disposition
    interval: IntegerInterval | None
    certificate: str | None = None
    reason: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.feature_id not in _FEATURE_SPEC:
            raise ObjectVersionSpaceError("scene feature is outside the frozen catalog")
        if not isinstance(self.disposition, Disposition):
            raise TypeError("scene feature disposition must be Disposition")
        certificate = _optional_text(self.certificate, "feature certificate")
        reason = _optional_text(self.reason, "feature reason")
        error_type = _optional_text(self.error_type, "feature error_type")
        if self.disposition is Disposition.PRESENT:
            if not isinstance(self.interval, IntegerInterval) or any(
                item is not None for item in (certificate, reason, error_type)
            ):
                raise ObjectVersionSpaceError(
                    "present scene feature requires only an integer interval"
                )
            maximum = _FEATURE_SPEC[self.feature_id].maximum
            if maximum is not None and self.interval.upper > maximum:
                raise ObjectVersionSpaceError("scene feature interval exceeds its unit")
        elif self.disposition is Disposition.CERTIFIED_ABSENT:
            if self.interval is not None or certificate is None or any(
                item is not None for item in (reason, error_type)
            ):
                raise ObjectVersionSpaceError(
                    "certified-absent scene feature requires only a certificate"
                )
        elif self.disposition is Disposition.INDETERMINATE:
            if self.interval is not None or reason is None or any(
                item is not None for item in (certificate, error_type)
            ):
                raise ObjectVersionSpaceError(
                    "indeterminate scene feature requires only a reason"
                )
        elif (
            self.interval is not None
            or reason is None
            or error_type is None
            or certificate is not None
        ):
            raise ObjectVersionSpaceError(
                "error scene feature requires a reason and error_type"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": OBJECT_SCENE_FEATURE_VALUE_SCHEMA,
            "feature_id": self.feature_id,
            "disposition": self.disposition.value,
            "interval": None if self.interval is None else self.interval.to_data(),
            "certificate": self.certificate,
            "reason": self.reason,
            "error_type": self.error_type,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneFeatureValue":
        raw = _fields(
            value,
            {
                "schema",
                "feature_id",
                "disposition",
                "interval",
                "certificate",
                "reason",
                "error_type",
            },
            "object scene feature value",
        )
        if raw["schema"] != OBJECT_SCENE_FEATURE_VALUE_SCHEMA:
            raise ObjectVersionSpaceError("scene feature schema differs")
        try:
            disposition = Disposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectVersionSpaceError("scene feature disposition is unknown") from exc
        result = cls(
            raw["feature_id"],
            disposition,
            None
            if raw["interval"] is None
            else IntegerInterval.from_data(raw["interval"]),
            raw["certificate"],
            raw["reason"],
            raw["error_type"],
        )
        if result.to_data() != dict(raw):
            raise ObjectVersionSpaceError("scene feature value is not canonical")
        return result


def _lineage_content(value: "ObjectStableLineageEvidence") -> dict[str, object]:
    return {
        "schema": OBJECT_STABLE_LINEAGE_EVIDENCE_SCHEMA,
        "lineage_id": value.lineage_id,
        "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
        "feature_values": [item.to_data() for item in value.feature_values],
        "same_object_across_segmentation_scenarios": True,
    }


@dataclass(frozen=True, slots=True)
class ObjectStableLineageEvidence:
    lineage_id: str
    feature_values: tuple[ObjectSceneFeatureValue, ...]
    lineage_digest: str

    def __post_init__(self) -> None:
        _code(self.lineage_id, "lineage_id")
        if (
            not isinstance(self.feature_values, tuple)
            or any(not isinstance(item, ObjectSceneFeatureValue) for item in self.feature_values)
            or tuple(item.feature_id for item in self.feature_values) != OBJECT_FEATURE_IDS
        ):
            raise ObjectVersionSpaceError(
                "lineage feature values must exhaust the frozen catalog in exact order"
            )
        _digest(self.lineage_digest, "lineage_digest")
        if self.lineage_digest != canonical_digest(_lineage_content(self)):
            raise ObjectVersionSpaceError("lineage digest differs")

    @classmethod
    def create(
        cls, lineage_id: str, feature_values: Sequence[ObjectSceneFeatureValue]
    ) -> "ObjectStableLineageEvidence":
        frozen = tuple(feature_values)
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "lineage_id", lineage_id)
        object.__setattr__(provisional, "feature_values", frozen)
        return cls(lineage_id, frozen, canonical_digest(_lineage_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_lineage_content(self), "lineage_digest": self.lineage_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectStableLineageEvidence":
        raw = _fields(
            value,
            {
                "schema",
                "lineage_id",
                "feature_catalog_digest",
                "feature_values",
                "same_object_across_segmentation_scenarios",
                "lineage_digest",
            },
            "stable lineage evidence",
        )
        if not isinstance(raw["feature_values"], list):
            raise ObjectVersionSpaceError("lineage feature_values must be a JSON list")
        result = cls(
            raw["lineage_id"],
            tuple(ObjectSceneFeatureValue.from_data(item) for item in raw["feature_values"]),
            raw["lineage_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectVersionSpaceError("lineage evidence is not canonical")
        return result


def _scene_content(value: "ObjectSceneEvidence") -> dict[str, object]:
    return {
        "schema": OBJECT_SCENE_EVIDENCE_SCHEMA,
        "scene_id": value.scene_id,
        "lineage_catalog_digest": value.lineage_catalog_digest,
        "feature_catalog_digest": OBJECT_FEATURE_CATALOG_DIGEST,
        "lineages": [item.to_data() for item in value.lineages],
        "unresolved_lineage_possible": value.unresolved_lineage_possible,
        "support_side_is_visual_evidence": False,
    }


@dataclass(frozen=True, slots=True)
class ObjectSceneEvidence:
    """Candidate-independent evidence; deliberately contains no side label."""

    scene_id: str
    lineage_catalog_digest: str
    lineages: tuple[ObjectStableLineageEvidence, ...]
    unresolved_lineage_possible: bool
    scene_digest: str

    def __post_init__(self) -> None:
        _scene_id(self.scene_id)
        _digest(self.lineage_catalog_digest, "lineage_catalog_digest")
        if (
            not isinstance(self.lineages, tuple)
            or any(not isinstance(item, ObjectStableLineageEvidence) for item in self.lineages)
            or tuple(item.lineage_id for item in self.lineages)
            != tuple(sorted(item.lineage_id for item in self.lineages))
            or len({item.lineage_id for item in self.lineages}) != len(self.lineages)
        ):
            raise ObjectVersionSpaceError("scene lineages must be unique and sorted")
        if type(self.unresolved_lineage_possible) is not bool:
            raise TypeError("unresolved_lineage_possible must be a literal bool")
        if not self.lineages and not self.unresolved_lineage_possible:
            raise ObjectVersionSpaceError(
                "an empty lineage inventory must remain explicitly unresolved"
            )
        _digest(self.scene_digest, "scene_digest")
        if self.scene_digest != canonical_digest(_scene_content(self)):
            raise ObjectVersionSpaceError("scene digest differs")

    @classmethod
    def create(
        cls,
        scene_id: str,
        lineage_catalog_digest: str,
        lineages: Sequence[ObjectStableLineageEvidence],
        *,
        unresolved_lineage_possible: bool,
    ) -> "ObjectSceneEvidence":
        frozen = tuple(lineages)
        provisional = object.__new__(cls)
        for name, item in (
            ("scene_id", scene_id),
            ("lineage_catalog_digest", lineage_catalog_digest),
            ("lineages", frozen),
            ("unresolved_lineage_possible", unresolved_lineage_possible),
        ):
            object.__setattr__(provisional, name, item)
        return cls(
            scene_id,
            lineage_catalog_digest,
            frozen,
            unresolved_lineage_possible,
            canonical_digest(_scene_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_scene_content(self), "scene_digest": self.scene_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSceneEvidence":
        raw = _fields(
            value,
            {
                "schema",
                "scene_id",
                "lineage_catalog_digest",
                "feature_catalog_digest",
                "lineages",
                "unresolved_lineage_possible",
                "support_side_is_visual_evidence",
                "scene_digest",
            },
            "object scene evidence",
        )
        if not isinstance(raw["lineages"], list):
            raise ObjectVersionSpaceError("scene lineages must be a JSON list")
        result = cls(
            raw["scene_id"],
            raw["lineage_catalog_digest"],
            tuple(ObjectStableLineageEvidence.from_data(item) for item in raw["lineages"]),
            raw["unresolved_lineage_possible"],
            raw["scene_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectVersionSpaceError("scene evidence is not canonical")
        return result


def _evaluate_atom_on_value(
    atom: ObjectProfileAtom, value: ObjectSceneFeatureValue
) -> Disposition:
    if atom.feature_id != value.feature_id:
        raise ObjectVersionSpaceError("atom and scene feature differ")
    if value.disposition is not Disposition.PRESENT:
        return value.disposition
    assert value.interval is not None
    if value.interval.lower >= atom.target:
        return Disposition.PRESENT
    if value.interval.upper < atom.target:
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def _combine_atoms(values: Sequence[Disposition]) -> Disposition:
    frozen = tuple(values)
    if not frozen:
        raise ObjectVersionSpaceError("candidate conjunction must be nonempty")
    if Disposition.CERTIFIED_ABSENT in frozen:
        return Disposition.CERTIFIED_ABSENT
    if all(item is Disposition.PRESENT for item in frozen):
        return Disposition.PRESENT
    if Disposition.ERROR in frozen:
        return Disposition.ERROR
    return Disposition.INDETERMINATE


@dataclass(frozen=True, slots=True)
class ObjectCandidateLineageEvaluation:
    lineage_id: str
    atom_dispositions: tuple[Disposition, ...]
    disposition: Disposition

    def __post_init__(self) -> None:
        _code(self.lineage_id, "evaluation lineage_id")
        if not self.atom_dispositions or any(
            not isinstance(item, Disposition) for item in self.atom_dispositions
        ):
            raise ObjectVersionSpaceError("lineage atom dispositions are malformed")
        if self.disposition is not _combine_atoms(self.atom_dispositions):
            raise ObjectVersionSpaceError("lineage disposition differs from atom replay")

    def to_data(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "atom_dispositions": [item.value for item in self.atom_dispositions],
            "disposition": self.disposition.value,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectCandidateLineageEvaluation":
        raw = _fields(
            value,
            {"lineage_id", "atom_dispositions", "disposition"},
            "candidate lineage evaluation",
        )
        if not isinstance(raw["atom_dispositions"], list):
            raise ObjectVersionSpaceError("atom_dispositions must be a JSON list")
        try:
            atoms = tuple(Disposition(item) for item in raw["atom_dispositions"])
            disposition = Disposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectVersionSpaceError("lineage evaluation disposition is unknown") from exc
        result = cls(raw["lineage_id"], atoms, disposition)
        if result.to_data() != dict(raw):
            raise ObjectVersionSpaceError("lineage evaluation is not canonical")
        return result


def _combine_lineages(
    values: Sequence[Disposition], *, unresolved_lineage_possible: bool
) -> Disposition:
    frozen = tuple(values)
    if Disposition.PRESENT in frozen:
        return Disposition.PRESENT
    if (
        frozen
        and
        not unresolved_lineage_possible
        and all(item is Disposition.CERTIFIED_ABSENT for item in frozen)
    ):
        return Disposition.CERTIFIED_ABSENT
    if Disposition.ERROR in frozen:
        return Disposition.ERROR
    return Disposition.INDETERMINATE


def _candidate_evaluation_content(
    value: "ObjectCandidateSceneEvaluation",
) -> dict[str, object]:
    return {
        "schema": OBJECT_CANDIDATE_EVALUATION_SCHEMA,
        "algorithm_digest": value.algorithm_digest,
        "grid_digest": value.grid_digest,
        "profile_digest": value.profile_digest,
        "scene_digest": value.scene_digest,
        "lineages": [item.to_data() for item in value.lineages],
        "unresolved_lineage_possible": value.unresolved_lineage_possible,
        "same_lineage_conjunction": True,
        "disposition": value.disposition.value,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectCandidateSceneEvaluation:
    algorithm_digest: str
    grid_digest: str
    profile_digest: str
    scene_digest: str
    lineages: tuple[ObjectCandidateLineageEvaluation, ...]
    unresolved_lineage_possible: bool
    disposition: Disposition
    evaluation_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != object_version_space_algorithm_digest():
            raise ObjectVersionSpaceError("evaluation algorithm binding differs")
        for name in (
            "grid_digest",
            "profile_digest",
            "scene_digest",
            "evaluation_digest",
        ):
            _digest(getattr(self, name), name)
        if (
            not isinstance(self.lineages, tuple)
            or any(
                not isinstance(item, ObjectCandidateLineageEvaluation)
                for item in self.lineages
            )
            or tuple(item.lineage_id for item in self.lineages)
            != tuple(sorted(item.lineage_id for item in self.lineages))
        ):
            raise ObjectVersionSpaceError("evaluation lineages are not canonical")
        if type(self.unresolved_lineage_possible) is not bool:
            raise TypeError("evaluation unresolved flag must be a literal bool")
        expected = _combine_lineages(
            tuple(item.disposition for item in self.lineages),
            unresolved_lineage_possible=self.unresolved_lineage_possible,
        )
        if self.disposition is not expected:
            raise ObjectVersionSpaceError("scene disposition differs from lineage replay")
        if self.evaluation_digest != canonical_digest(
            _candidate_evaluation_content(self)
        ):
            raise ObjectVersionSpaceError("candidate evaluation digest differs")

    def to_data(self) -> dict[str, object]:
        return {
            **_candidate_evaluation_content(self),
            "evaluation_digest": self.evaluation_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectCandidateSceneEvaluation":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_digest",
                "grid_digest",
                "profile_digest",
                "scene_digest",
                "lineages",
                "unresolved_lineage_possible",
                "same_lineage_conjunction",
                "disposition",
                *_authority_data(),
                "evaluation_digest",
            },
            "object candidate scene evaluation",
        )
        if not isinstance(raw["lineages"], list):
            raise ObjectVersionSpaceError("evaluation lineages must be a JSON list")
        try:
            disposition = Disposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectVersionSpaceError("evaluation disposition is unknown") from exc
        result = cls(
            raw["algorithm_digest"],
            raw["grid_digest"],
            raw["profile_digest"],
            raw["scene_digest"],
            tuple(
                ObjectCandidateLineageEvaluation.from_data(item)
                for item in raw["lineages"]
            ),
            raw["unresolved_lineage_possible"],
            disposition,
            raw["evaluation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectVersionSpaceError("candidate evaluation is not canonical")
        return result


def evaluate_object_profile_candidate(
    grid: ObjectPredicateGrid,
    profile: ObjectProfile,
    scene: ObjectSceneEvidence,
) -> ObjectCandidateSceneEvaluation:
    """Evaluate a profile without mixing its atoms across object lineages."""

    grid = ObjectPredicateGrid.from_data(grid.to_data())
    profile = ObjectProfile.from_data(profile.to_data())
    scene = ObjectSceneEvidence.from_data(scene.to_data())
    if profile not in enumerate_object_profile_candidates(grid):
        raise ObjectVersionSpaceError("profile is outside the complete task grid")
    return _evaluate_validated_candidate(grid, profile, scene)


def _evaluate_validated_candidate(
    grid: ObjectPredicateGrid,
    profile: ObjectProfile,
    scene: ObjectSceneEvidence,
) -> ObjectCandidateSceneEvaluation:
    lineages: list[ObjectCandidateLineageEvaluation] = []
    for lineage in scene.lineages:
        by_feature = {item.feature_id: item for item in lineage.feature_values}
        atom_dispositions = tuple(
            _evaluate_atom_on_value(atom, by_feature[atom.feature_id])
            for atom in profile.atoms
        )
        lineages.append(
            ObjectCandidateLineageEvaluation(
                lineage.lineage_id,
                atom_dispositions,
                _combine_atoms(atom_dispositions),
            )
        )
    frozen_lineages = tuple(lineages)
    disposition = _combine_lineages(
        tuple(item.disposition for item in frozen_lineages),
        unresolved_lineage_possible=scene.unresolved_lineage_possible,
    )
    values = {
        "algorithm_digest": grid.algorithm_digest,
        "grid_digest": grid.grid_digest,
        "profile_digest": profile.profile_digest,
        "scene_digest": scene.scene_digest,
        "lineages": frozen_lineages,
        "unresolved_lineage_possible": scene.unresolved_lineage_possible,
        "disposition": disposition,
    }
    provisional = object.__new__(ObjectCandidateSceneEvaluation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectCandidateSceneEvaluation(
        **values,  # type: ignore[arg-type]
        evaluation_digest=canonical_digest(_candidate_evaluation_content(provisional)),
    )


@dataclass(frozen=True, slots=True)
class ObjectSupportDiagnostic:
    profile_digest: str
    definite_counterexample_scene_ids: tuple[str, ...]
    indeterminate_scene_ids: tuple[str, ...]
    error_scene_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.profile_digest, "diagnostic profile_digest")
        for name in (
            "definite_counterexample_scene_ids",
            "indeterminate_scene_ids",
            "error_scene_ids",
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or any(_SCENE_ID.fullmatch(item) is None for item in values)
                or len(values) != len(set(values))
            ):
                raise ObjectVersionSpaceError(f"{name} is not canonical")

    def to_data(self) -> dict[str, object]:
        return {
            "schema": OBJECT_SUPPORT_DIAGNOSTIC_SCHEMA,
            "profile_digest": self.profile_digest,
            "definite_counterexample_scene_ids": list(
                self.definite_counterexample_scene_ids
            ),
            "indeterminate_scene_ids": list(self.indeterminate_scene_ids),
            "error_scene_ids": list(self.error_scene_ids),
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSupportDiagnostic":
        raw = _fields(
            value,
            {
                "schema",
                "profile_digest",
                "definite_counterexample_scene_ids",
                "indeterminate_scene_ids",
                "error_scene_ids",
            },
            "object support diagnostic",
        )
        result = cls(
            raw["profile_digest"],
            tuple(raw["definite_counterexample_scene_ids"]),
            tuple(raw["indeterminate_scene_ids"]),
            tuple(raw["error_scene_ids"]),
        )
        if result.to_data() != dict(raw):
            raise ObjectVersionSpaceError("support diagnostic is not canonical")
        return result


def _gap_content(value: "ObjectSupportGap") -> dict[str, object]:
    return {
        "schema": OBJECT_SUPPORT_GAP_SCHEMA,
        "kind": value.kind.value,
        "diagnostics": [item.to_data() for item in value.diagnostics],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSupportGap:
    kind: ObjectSupportGapKind
    diagnostics: tuple[ObjectSupportDiagnostic, ...]
    gap_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ObjectSupportGapKind):
            raise TypeError("gap kind must be ObjectSupportGapKind")
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, ObjectSupportDiagnostic) for item in self.diagnostics
        ):
            raise ObjectVersionSpaceError("gap diagnostics are malformed")
        _digest(self.gap_digest, "gap_digest")
        if self.gap_digest != canonical_digest(_gap_content(self)):
            raise ObjectVersionSpaceError("gap digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectSupportGap":
        raw = _fields(
            value,
            {"schema", "kind", "diagnostics", *_authority_data(), "gap_digest"},
            "object support gap",
        )
        if not isinstance(raw["diagnostics"], list):
            raise ObjectVersionSpaceError("gap diagnostics must be a JSON list")
        try:
            kind = ObjectSupportGapKind(raw["kind"])
        except (TypeError, ValueError) as exc:
            raise ObjectVersionSpaceError("support gap kind is unknown") from exc
        result = cls(
            kind,
            tuple(ObjectSupportDiagnostic.from_data(item) for item in raw["diagnostics"]),
            raw["gap_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectVersionSpaceError("support gap is not canonical")
        return result


class ObjectSupportSide(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


def _is_survivor(
    row: Sequence[Disposition], sides: Sequence[ObjectSupportSide]
) -> bool:
    return all(
        state
        is (
            Disposition.PRESENT
            if side is ObjectSupportSide.POSITIVE
            else Disposition.CERTIFIED_ABSENT
        )
        for state, side in zip(row, sides, strict=True)
    )


def _support_diagnostics(
    candidates: Sequence[ObjectProfile],
    scene_ids: Sequence[str],
    sides: Sequence[ObjectSupportSide],
    rows: Sequence[Sequence[Disposition]],
) -> tuple[ObjectSupportDiagnostic, ...]:
    result: list[ObjectSupportDiagnostic] = []
    for candidate, row in zip(candidates, rows, strict=True):
        definite = tuple(
            scene_id
            for scene_id, side, state in zip(scene_ids, sides, row, strict=True)
            if (
                side is ObjectSupportSide.POSITIVE
                and state is Disposition.CERTIFIED_ABSENT
            )
            or (
                side is ObjectSupportSide.NEGATIVE
                and state is Disposition.PRESENT
            )
        )
        result.append(
            ObjectSupportDiagnostic(
                candidate.profile_digest,
                definite,
                tuple(
                    scene_id
                    for scene_id, state in zip(scene_ids, row, strict=True)
                    if state is Disposition.INDETERMINATE
                ),
                tuple(
                    scene_id
                    for scene_id, state in zip(scene_ids, row, strict=True)
                    if state is Disposition.ERROR
                ),
            )
        )
    return tuple(result)


def _make_support_gap(
    candidates: Sequence[ObjectProfile],
    scene_ids: Sequence[str],
    sides: Sequence[ObjectSupportSide],
    rows: Sequence[Sequence[Disposition]],
) -> ObjectSupportGap:
    diagnostics = _support_diagnostics(candidates, scene_ids, sides, rows)
    witness_needed = any(
        not item.definite_counterexample_scene_ids
        and bool(item.indeterminate_scene_ids or item.error_scene_ids)
        for item in diagnostics
    )
    kind = (
        ObjectSupportGapKind.WITNESS_GAP
        if witness_needed
        else ObjectSupportGapKind.LANGUAGE_GAP
    )
    provisional = object.__new__(ObjectSupportGap)
    object.__setattr__(provisional, "kind", kind)
    object.__setattr__(provisional, "diagnostics", diagnostics)
    return ObjectSupportGap(kind, diagnostics, canonical_digest(_gap_content(provisional)))


def _version_content(value: "ObjectSupportVersionSpace") -> dict[str, object]:
    return {
        "schema": OBJECT_SUPPORT_VERSION_SPACE_SCHEMA,
        "algorithm_id": OBJECT_VERSION_SPACE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "grid": value.grid.to_data(),
        "candidates": [item.to_data() for item in value.candidates],
        "support_scene_ids": list(value.support_scene_ids),
        "support_scene_digests": list(value.support_scene_digests),
        "support_sides": [item.value for item in value.support_sides],
        "rows": [[state.value for state in row] for row in value.rows],
        "survivor_profile_digests": list(value.survivor_profile_digests),
        "gap": None if value.gap is None else value.gap.to_data(),
        "positive_accept": Disposition.PRESENT.value,
        "negative_accept": Disposition.CERTIFIED_ABSENT.value,
        "same_lineage_conjunction": True,
        "complete_finite_inventory": True,
        "codex_may_rank_survivors_only": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectSupportVersionSpace:
    algorithm_digest: str
    grid: ObjectPredicateGrid
    candidates: tuple[ObjectProfile, ...]
    support_scene_ids: tuple[str, ...]
    support_scene_digests: tuple[str, ...]
    support_sides: tuple[ObjectSupportSide, ...]
    rows: tuple[tuple[Disposition, ...], ...]
    survivor_profile_digests: tuple[str, ...]
    gap: ObjectSupportGap | None
    version_space_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != object_version_space_algorithm_digest():
            raise ObjectVersionSpaceError("version-space algorithm binding differs")
        if not isinstance(self.grid, ObjectPredicateGrid):
            raise TypeError("version-space grid must be ObjectPredicateGrid")
        expected_candidates = enumerate_object_profile_candidates(self.grid)
        if self.candidates != expected_candidates:
            raise ObjectVersionSpaceError("candidate inventory is not complete and canonical")
        panel_count = len(self.support_scene_ids)
        if (
            panel_count < 2
            or len(set(self.support_scene_ids)) != panel_count
            or any(_SCENE_ID.fullmatch(item) is None for item in self.support_scene_ids)
            or len(self.support_scene_digests) != panel_count
            or len(self.support_sides) != panel_count
            or ObjectSupportSide.POSITIVE not in self.support_sides
            or ObjectSupportSide.NEGATIVE not in self.support_sides
            or self.support_sides
            != tuple(sorted(self.support_sides, key=lambda item: item.value, reverse=True))
        ):
            raise ObjectVersionSpaceError("support scene inventory is not canonical")
        for item in self.support_scene_digests:
            _digest(item, "support scene digest")
        if (
            len(self.rows) != len(self.candidates)
            or any(len(row) != panel_count for row in self.rows)
            or any(
                not isinstance(state, Disposition)
                for row in self.rows
                for state in row
            )
        ):
            raise ObjectVersionSpaceError("version-space rows are malformed")
        expected_survivors = tuple(
            candidate.profile_digest
            for candidate, row in zip(self.candidates, self.rows, strict=True)
            if _is_survivor(row, self.support_sides)
        )
        if self.survivor_profile_digests != expected_survivors:
            raise ObjectVersionSpaceError("survivor inventory differs from exact replay")
        expected_gap = (
            None
            if expected_survivors
            else _make_support_gap(
                self.candidates,
                self.support_scene_ids,
                self.support_sides,
                self.rows,
            )
        )
        if self.gap != expected_gap:
            raise ObjectVersionSpaceError("typed support gap differs from exact replay")
        _digest(self.version_space_digest, "version_space_digest")
        if self.version_space_digest != canonical_digest(_version_content(self)):
            raise ObjectVersionSpaceError("version-space digest differs")

    def row(self, profile_digest: str) -> tuple[Disposition, ...]:
        _digest(profile_digest, "row profile_digest")
        matches = tuple(
            row
            for candidate, row in zip(self.candidates, self.rows, strict=True)
            if candidate.profile_digest == profile_digest
        )
        if len(matches) != 1:
            raise ObjectVersionSpaceError("candidate row is absent")
        return matches[0]

    def survivor(self, profile_digest: str) -> ObjectProfile:
        if profile_digest not in self.survivor_profile_digests:
            raise ObjectVersionSpaceError("profile is not a verified survivor")
        return next(
            item for item in self.candidates if item.profile_digest == profile_digest
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_version_content(self),
            "version_space_digest": self.version_space_digest,
        }

    @classmethod
    def from_data(cls, value: object) -> "ObjectSupportVersionSpace":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_id",
                "algorithm_digest",
                "grid",
                "candidates",
                "support_scene_ids",
                "support_scene_digests",
                "support_sides",
                "rows",
                "survivor_profile_digests",
                "gap",
                "positive_accept",
                "negative_accept",
                "same_lineage_conjunction",
                "complete_finite_inventory",
                "codex_may_rank_survivors_only",
                *_authority_data(),
                "version_space_digest",
            },
            "object support version space",
        )
        for name in (
            "candidates",
            "support_scene_ids",
            "support_scene_digests",
            "support_sides",
            "rows",
            "survivor_profile_digests",
        ):
            if not isinstance(raw[name], list):
                raise ObjectVersionSpaceError(f"{name} must be a JSON list")
        try:
            sides = tuple(ObjectSupportSide(item) for item in raw["support_sides"])
            rows = tuple(
                tuple(Disposition(item) for item in row) for row in raw["rows"]
            )
        except (TypeError, ValueError) as exc:
            raise ObjectVersionSpaceError("support side or row disposition is unknown") from exc
        gap_raw = raw["gap"]
        result = cls(
            raw["algorithm_digest"],
            ObjectPredicateGrid.from_data(raw["grid"]),
            tuple(ObjectProfile.from_data(item) for item in raw["candidates"]),
            tuple(raw["support_scene_ids"]),
            tuple(raw["support_scene_digests"]),
            sides,
            rows,
            tuple(raw["survivor_profile_digests"]),
            None if gap_raw is None else ObjectSupportGap.from_data(gap_raw),
            raw["version_space_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectVersionSpaceError("version space is not canonical")
        return result


def build_object_support_version_space(
    grid: ObjectPredicateGrid,
    positives: Sequence[ObjectSceneEvidence],
    negatives: Sequence[ObjectSceneEvidence],
) -> ObjectSupportVersionSpace:
    """Build the complete support-consistent version space.

    The expected side is supplied here and is absent from every visual evidence
    record.  Scenes are canonicalized by side, then by ``scene_id``.
    """

    grid = ObjectPredicateGrid.from_data(grid.to_data())
    positive_scenes = tuple(
        sorted(
            (ObjectSceneEvidence.from_data(item.to_data()) for item in positives),
            key=lambda item: item.scene_id,
        )
    )
    negative_scenes = tuple(
        sorted(
            (ObjectSceneEvidence.from_data(item.to_data()) for item in negatives),
            key=lambda item: item.scene_id,
        )
    )
    if not positive_scenes or not negative_scenes:
        raise ObjectVersionSpaceError("support requires at least one scene per side")
    scenes = positive_scenes + negative_scenes
    if len({item.scene_id for item in scenes}) != len(scenes):
        raise ObjectVersionSpaceError("support scene IDs must be globally unique")
    sides = (ObjectSupportSide.POSITIVE,) * len(positive_scenes) + (
        ObjectSupportSide.NEGATIVE,
    ) * len(negative_scenes)
    candidates = enumerate_object_profile_candidates(grid)
    rows = tuple(
        tuple(
            _evaluate_validated_candidate(grid, candidate, scene).disposition
            for scene in scenes
        )
        for candidate in candidates
    )
    survivors = tuple(
        candidate.profile_digest
        for candidate, row in zip(candidates, rows, strict=True)
        if _is_survivor(row, sides)
    )
    gap = (
        None
        if survivors
        else _make_support_gap(
            candidates,
            tuple(item.scene_id for item in scenes),
            sides,
            rows,
        )
    )
    values = {
        "algorithm_digest": grid.algorithm_digest,
        "grid": grid,
        "candidates": candidates,
        "support_scene_ids": tuple(item.scene_id for item in scenes),
        "support_scene_digests": tuple(item.scene_digest for item in scenes),
        "support_sides": sides,
        "rows": rows,
        "survivor_profile_digests": survivors,
        "gap": gap,
    }
    provisional = object.__new__(ObjectSupportVersionSpace)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectSupportVersionSpace(
        **values,  # type: ignore[arg-type]
        version_space_digest=canonical_digest(_version_content(provisional)),
    )


def cold_verify_object_support_version_space(
    version_space: ObjectSupportVersionSpace,
    grid: ObjectPredicateGrid,
    positives: Sequence[ObjectSceneEvidence],
    negatives: Sequence[ObjectSceneEvidence],
) -> ObjectSupportVersionSpace:
    """Model-free replay of candidate enumeration, decisions, and gap typing."""

    decoded = ObjectSupportVersionSpace.from_data(version_space.to_data())
    replay = build_object_support_version_space(grid, positives, negatives)
    if decoded != replay:
        raise ObjectVersionSpaceError("cold version-space replay differs")
    return decoded


__all__ = (
    "COUNT_THRESHOLDS",
    "MAX_CONJUNCTION_ATOMS",
    "OBJECT_VERSION_SPACE_ALGORITHM_ID",
    "ObjectCandidateLineageEvaluation",
    "ObjectCandidateSceneEvaluation",
    "ObjectPredicateGrid",
    "ObjectSceneEvidence",
    "ObjectSceneFeatureValue",
    "ObjectStableLineageEvidence",
    "ObjectSupportDiagnostic",
    "ObjectSupportGap",
    "ObjectSupportGapKind",
    "ObjectSupportSide",
    "ObjectSupportVersionSpace",
    "ObjectVersionSpaceError",
    "PPM_THRESHOLDS",
    "build_object_support_version_space",
    "cold_verify_object_support_version_space",
    "enumerate_object_profile_candidates",
    "evaluate_object_profile_candidate",
    "object_version_space_algorithm_digest",
)
