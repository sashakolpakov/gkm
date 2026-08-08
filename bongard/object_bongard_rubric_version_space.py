"""Closed pure-Python predicates over ordinal visual-rubric observations.

The vision observer supplies candidate-independent ordinal preference
intervals.  This module is the only decision authority: for one frozen rubric
it enumerates exactly two positive ``AT_LEAST 3`` predicates, one per scope.
Presence requires a lower bound of at least 3, absence requires an upper bound
of at most 1, and level 2 is a mandatory deadband.  There is no negation,
polarity repair, disjunction, arbitrary code, or model-selected threshold.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricObserverArtifact,
    ObjectBongardRubricSpec,
    RubricObservationState,
    RubricScope,
    RubricScopeObservation,
    object_bongard_rubric_ordinal_scale_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


RUBRIC_CANDIDATE_SCHEMA = "gkm.bongard-object-rubric-candidate.v2"
RUBRIC_CANDIDATE_EVALUATION_SCHEMA = (
    "gkm.bongard-object-rubric-candidate-evaluation.v2"
)
RUBRIC_SUPPORT_DIAGNOSTIC_SCHEMA = (
    "gkm.bongard-object-rubric-support-diagnostic.v1"
)
RUBRIC_SUPPORT_GAP_SCHEMA = "gkm.bongard-object-rubric-support-gap.v1"
RUBRIC_SUPPORT_VERSION_SPACE_SCHEMA = (
    "gkm.bongard-object-rubric-support-version-space.v2"
)
RUBRIC_VERSION_SPACE_ALGORITHM_ID = (
    "bongard.object-rubric-version-space/positive-prefers-deadband-v2"
)
RUBRIC_PRESENT_LOWER_BOUND = 3
RUBRIC_ABSENCE_UPPER_BOUND = 1
RUBRIC_THRESHOLDS = (RUBRIC_PRESENT_LOWER_BOUND,)
RUBRIC_SUPPORT_PANELS_PER_SIDE = 6

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")


class ObjectBongardRubricVersionSpaceError(ValueError):
    """A candidate, evaluation, or support inventory is malformed."""


class RubricPredicateOperator(str, Enum):
    AT_LEAST = "at_least"


class RubricSupportSide(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class RubricSupportGapKind(str, Enum):
    LANGUAGE_GAP = "language_gap"
    WITNESS_GAP = "witness_gap"


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
        "lean_removal_changes_decision": False,
    }


def _language_data() -> dict[str, object]:
    return {
        "operator": RubricPredicateOperator.AT_LEAST.value,
        "thresholds": list(RUBRIC_THRESHOLDS),
        "present_when_interval_lower_at_least": RUBRIC_PRESENT_LOWER_BOUND,
        "certified_absent_when_interval_upper_at_most": (
            RUBRIC_ABSENCE_UPPER_BOUND
        ),
        "deadband_levels": [2],
        "deadband_disposition": Disposition.INDETERMINATE.value,
        "tie_can_certify_absence": False,
        "ordinal_scale_digest": object_bongard_rubric_ordinal_scale_digest(),
        "scopes": [RubricScope.OBJECT.value, RubricScope.SCENE.value],
        "object_quantifier": "exists-over-stable-object-observations",
        "unresolved_objects_can_prove_presence": False,
        "unresolved_objects_can_block_absence": True,
        "scene_source": "exactly-one-optional-canonical-scene-observation",
        "negation": False,
        "polarity_flip": False,
        "disjunction": False,
        "arbitrary_code": False,
        "model_chosen_threshold": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardRubricVersionSpaceError(
            f"{label} fields differ from the closed schema"
        )
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardRubricVersionSpaceError(
            f"{label} must be a lowercase raw SHA-256"
        )
    return value


def _panel_id(value: object, label: str = "panel_id") -> str:
    if not isinstance(value, str) or _PANEL_ID.fullmatch(value) is None:
        raise ObjectBongardRubricVersionSpaceError(
            f"{label} must be a bounded panel identifier"
        )
    return value


def _threshold(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in (
        RUBRIC_THRESHOLDS
    ):
        raise ObjectBongardRubricVersionSpaceError(
            "rubric threshold must be the frozen target-preference level"
        )
    return value


def object_bongard_rubric_version_space_source_digest() -> str:
    """Return the digest only while the loaded evaluator source is unchanged."""

    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_bongard_rubric_version_space_algorithm_digest() -> str:
    """Digest the exact loaded Python evaluator and its closed language."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-version-space-algorithm.v2",
            "algorithm_id": RUBRIC_VERSION_SPACE_ALGORITHM_ID,
            "implementation_source_sha256": (
                object_bongard_rubric_version_space_source_digest()
            ),
            "language": _language_data(),
            "positive_accept": Disposition.PRESENT.value,
            "negative_accept": Disposition.CERTIFIED_ABSENT.value,
            "support_panels_per_side": RUBRIC_SUPPORT_PANELS_PER_SIDE,
            "candidate_order": "object-prefers-target-then-scene-prefers-target",
            **_authority_data(),
        }
    )


def _candidate_id(scope: RubricScope, threshold: int) -> str:
    return f"rubric:{scope.value}:at_least:{threshold}"


def _formula(scope: RubricScope, threshold: int) -> str:
    if scope is RubricScope.OBJECT:
        return f"EXISTS OBJECT witness with rubric_level >= {threshold}"
    return f"SCENE rubric_level >= {threshold}"


def _candidate_content(value: "ObjectBongardRubricCandidate") -> dict[str, object]:
    return {
        "schema": RUBRIC_CANDIDATE_SCHEMA,
        "algorithm_id": RUBRIC_VERSION_SPACE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "candidate_id": value.candidate_id,
        "scope": value.scope.value,
        "operator": RubricPredicateOperator.AT_LEAST.value,
        "threshold": value.threshold,
        "formula": value.formula,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCandidate:
    """One member of the fixed two-scope target-preference inventory."""

    rubric_spec_digest: str
    scope: RubricScope
    threshold: int
    algorithm_digest: str
    candidate_digest: str

    def __post_init__(self) -> None:
        _digest(self.rubric_spec_digest, "rubric_spec_digest")
        if not isinstance(self.scope, RubricScope):
            raise TypeError("candidate scope must be RubricScope")
        _threshold(self.threshold)
        if self.algorithm_digest != (
            object_bongard_rubric_version_space_algorithm_digest()
        ):
            raise ObjectBongardRubricVersionSpaceError(
                "candidate algorithm binding differs"
            )
        _digest(self.candidate_digest, "candidate_digest")
        if self.candidate_digest != canonical_digest(_candidate_content(self)):
            raise ObjectBongardRubricVersionSpaceError(
                "candidate digest differs from canonical content"
            )

    @property
    def candidate_id(self) -> str:
        return _candidate_id(self.scope, self.threshold)

    @property
    def operator(self) -> RubricPredicateOperator:
        return RubricPredicateOperator.AT_LEAST

    @property
    def formula(self) -> str:
        return _formula(self.scope, self.threshold)

    @classmethod
    def create(
        cls, rubric_spec_digest: str, scope: RubricScope, threshold: int
    ) -> "ObjectBongardRubricCandidate":
        algorithm_digest = object_bongard_rubric_version_space_algorithm_digest()
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "rubric_spec_digest", rubric_spec_digest)
        object.__setattr__(provisional, "scope", scope)
        object.__setattr__(provisional, "threshold", threshold)
        object.__setattr__(provisional, "algorithm_digest", algorithm_digest)
        return cls(
            rubric_spec_digest,
            scope,
            threshold,
            algorithm_digest,
            canonical_digest(_candidate_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_candidate_content(self), "candidate_digest": self.candidate_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricCandidate":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_id",
                "algorithm_digest",
                "rubric_spec_digest",
                "candidate_id",
                "scope",
                "operator",
                "threshold",
                "formula",
                *_authority_data(),
                "candidate_digest",
            },
            "rubric candidate",
        )
        try:
            scope = RubricScope(raw["scope"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardRubricVersionSpaceError(
                "candidate scope is unknown"
            ) from exc
        result = cls(
            raw["rubric_spec_digest"],
            scope,
            raw["threshold"],
            raw["algorithm_digest"],
            raw["candidate_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricVersionSpaceError(
                "rubric candidate is not canonical"
            )
        return result


def enumerate_object_bongard_rubric_candidates(
    spec: ObjectBongardRubricSpec,
) -> tuple[ObjectBongardRubricCandidate, ...]:
    """Return the complete deterministic object/scene candidate inventory."""

    if not isinstance(spec, ObjectBongardRubricSpec):
        raise TypeError("spec must be ObjectBongardRubricSpec")
    _digest(spec.spec_digest, "rubric spec digest")
    return tuple(
        ObjectBongardRubricCandidate.create(spec.spec_digest, scope, threshold)
        for scope in (RubricScope.OBJECT, RubricScope.SCENE)
        for threshold in RUBRIC_THRESHOLDS
    )


# Short names are useful to callers while the longer names make artifact
# ownership explicit in serialized APIs.
RubricPredicateCandidate = ObjectBongardRubricCandidate


def _evaluation_content(
    value: "ObjectBongardRubricCandidateEvaluation",
) -> dict[str, object]:
    return {
        "schema": RUBRIC_CANDIDATE_EVALUATION_SCHEMA,
        "algorithm_id": RUBRIC_VERSION_SPACE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "candidate_digest": value.candidate_digest,
        "observer_artifact_digest": value.observer_artifact_digest,
        "panel_id": value.panel_id,
        "disposition": value.disposition.value,
        "positive_witness_source": (
            "stable-object-only-for-object-scope;canonical-scene-only-for-scene-scope"
        ),
        "failed_or_missing_observation_is_absence": False,
        "deadband_disposition_policy": (
            "present-lower-ge-3-absent-upper-le-1-else-indeterminate"
        ),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricCandidateEvaluation:
    """One candidate's typed disposition on one side-free panel artifact."""

    algorithm_digest: str
    rubric_spec_digest: str
    candidate_digest: str
    observer_artifact_digest: str
    panel_id: str
    disposition: Disposition
    evaluation_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != (
            object_bongard_rubric_version_space_algorithm_digest()
        ):
            raise ObjectBongardRubricVersionSpaceError(
                "evaluation algorithm binding differs"
            )
        for name in (
            "rubric_spec_digest",
            "candidate_digest",
            "observer_artifact_digest",
            "evaluation_digest",
        ):
            _digest(getattr(self, name), name)
        _panel_id(self.panel_id)
        if not isinstance(self.disposition, Disposition):
            raise TypeError("evaluation disposition must be Disposition")
        admitted = {
            item.candidate_digest
            for item in _enumerate_for_digest(self.rubric_spec_digest)
        }
        if self.candidate_digest not in admitted:
            raise ObjectBongardRubricVersionSpaceError(
                "evaluation candidate is outside the closed two-member inventory"
            )
        if self.evaluation_digest != canonical_digest(_evaluation_content(self)):
            raise ObjectBongardRubricVersionSpaceError(
                "evaluation digest differs from canonical content"
            )

    def to_data(self) -> dict[str, object]:
        return {
            **_evaluation_content(self),
            "evaluation_digest": self.evaluation_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricCandidateEvaluation":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_id",
                "algorithm_digest",
                "rubric_spec_digest",
                "candidate_digest",
                "observer_artifact_digest",
                "panel_id",
                "disposition",
                "positive_witness_source",
                "failed_or_missing_observation_is_absence",
                "deadband_disposition_policy",
                *_authority_data(),
                "evaluation_digest",
            },
            "rubric candidate evaluation",
        )
        try:
            disposition = Disposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardRubricVersionSpaceError(
                "evaluation disposition is unknown"
            ) from exc
        result = cls(
            raw["algorithm_digest"],
            raw["rubric_spec_digest"],
            raw["candidate_digest"],
            raw["observer_artifact_digest"],
            raw["panel_id"],
            disposition,
            raw["evaluation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricVersionSpaceError(
                "rubric candidate evaluation is not canonical"
            )
        return result


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricSupportDiagnostic:
    candidate_digest: str
    definite_counterexample_panel_ids: tuple[str, ...]
    indeterminate_panel_ids: tuple[str, ...]
    error_panel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.candidate_digest, "diagnostic candidate_digest")
        for name in (
            "definite_counterexample_panel_ids",
            "indeterminate_panel_ids",
            "error_panel_ids",
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or any(_PANEL_ID.fullmatch(item) is None for item in values)
                or values != tuple(sorted(set(values)))
            ):
                raise ObjectBongardRubricVersionSpaceError(
                    f"{name} is not a canonical panel inventory"
                )
        inventories = (
            set(self.definite_counterexample_panel_ids),
            set(self.indeterminate_panel_ids),
            set(self.error_panel_ids),
        )
        if any(
            inventories[left] & inventories[right]
            for left in range(len(inventories))
            for right in range(left + 1, len(inventories))
        ):
            raise ObjectBongardRubricVersionSpaceError(
                "diagnostic panel categories must be pairwise disjoint"
            )

    def to_data(self) -> dict[str, object]:
        return {
            "schema": RUBRIC_SUPPORT_DIAGNOSTIC_SCHEMA,
            "candidate_digest": self.candidate_digest,
            "definite_counterexample_panel_ids": list(
                self.definite_counterexample_panel_ids
            ),
            "indeterminate_panel_ids": list(self.indeterminate_panel_ids),
            "error_panel_ids": list(self.error_panel_ids),
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricSupportDiagnostic":
        raw = _fields(
            value,
            {
                "schema",
                "candidate_digest",
                "definite_counterexample_panel_ids",
                "indeterminate_panel_ids",
                "error_panel_ids",
            },
            "rubric support diagnostic",
        )
        for name in (
            "definite_counterexample_panel_ids",
            "indeterminate_panel_ids",
            "error_panel_ids",
        ):
            if not isinstance(raw[name], list):
                raise ObjectBongardRubricVersionSpaceError(
                    f"{name} must be a JSON list"
                )
        result = cls(
            raw["candidate_digest"],
            tuple(raw["definite_counterexample_panel_ids"]),
            tuple(raw["indeterminate_panel_ids"]),
            tuple(raw["error_panel_ids"]),
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricVersionSpaceError(
                "rubric support diagnostic is not canonical"
            )
        return result


def _gap_content(value: "ObjectBongardRubricSupportGap") -> dict[str, object]:
    return {
        "schema": RUBRIC_SUPPORT_GAP_SCHEMA,
        "kind": value.kind.value,
        "diagnostics": [item.to_data() for item in value.diagnostics],
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricSupportGap:
    kind: RubricSupportGapKind
    diagnostics: tuple[ObjectBongardRubricSupportDiagnostic, ...]
    gap_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RubricSupportGapKind):
            raise TypeError("gap kind must be RubricSupportGapKind")
        if not isinstance(self.diagnostics, tuple) or any(
            not isinstance(item, ObjectBongardRubricSupportDiagnostic)
            for item in self.diagnostics
        ):
            raise ObjectBongardRubricVersionSpaceError(
                "gap diagnostics are malformed"
            )
        if (
            len(self.diagnostics) != len(RUBRIC_THRESHOLDS) * 2
            or len({item.candidate_digest for item in self.diagnostics})
            != len(self.diagnostics)
            or tuple(item.candidate_digest for item in self.diagnostics)
            != tuple(sorted(item.candidate_digest for item in self.diagnostics))
        ):
            raise ObjectBongardRubricVersionSpaceError(
                "gap diagnostics must cover every unique digest-ordered candidate"
            )
        _digest(self.gap_digest, "gap_digest")
        if self.gap_digest != canonical_digest(_gap_content(self)):
            raise ObjectBongardRubricVersionSpaceError(
                "support gap digest differs from canonical content"
            )

    def to_data(self) -> dict[str, object]:
        return {**_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricSupportGap":
        raw = _fields(
            value,
            {"schema", "kind", "diagnostics", *_authority_data(), "gap_digest"},
            "rubric support gap",
        )
        if not isinstance(raw["diagnostics"], list):
            raise ObjectBongardRubricVersionSpaceError(
                "gap diagnostics must be a JSON list"
            )
        try:
            kind = RubricSupportGapKind(raw["kind"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardRubricVersionSpaceError(
                "support gap kind is unknown"
            ) from exc
        result = cls(
            kind,
            tuple(
                ObjectBongardRubricSupportDiagnostic.from_data(item)
                for item in raw["diagnostics"]
            ),
            raw["gap_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricVersionSpaceError(
                "rubric support gap is not canonical"
            )
        return result


def _enumerate_for_digest(
    rubric_spec_digest: str,
) -> tuple[ObjectBongardRubricCandidate, ...]:
    _digest(rubric_spec_digest, "rubric_spec_digest")
    return tuple(
        ObjectBongardRubricCandidate.create(rubric_spec_digest, scope, threshold)
        for scope in (RubricScope.OBJECT, RubricScope.SCENE)
        for threshold in RUBRIC_THRESHOLDS
    )


def _is_survivor(
    row: Sequence[Disposition], sides: Sequence[RubricSupportSide]
) -> bool:
    return all(
        state
        is (
            Disposition.PRESENT
            if side is RubricSupportSide.POSITIVE
            else Disposition.CERTIFIED_ABSENT
        )
        for state, side in zip(row, sides, strict=True)
    )


def _support_diagnostics(
    candidates: Sequence[ObjectBongardRubricCandidate],
    panel_ids: Sequence[str],
    sides: Sequence[RubricSupportSide],
    rows: Sequence[Sequence[Disposition]],
) -> tuple[ObjectBongardRubricSupportDiagnostic, ...]:
    result: list[ObjectBongardRubricSupportDiagnostic] = []
    for candidate, row in zip(candidates, rows, strict=True):
        definite = tuple(
            panel_id
            for panel_id, side, state in zip(panel_ids, sides, row, strict=True)
            if (
                side is RubricSupportSide.POSITIVE
                and state is Disposition.CERTIFIED_ABSENT
            )
            or (
                side is RubricSupportSide.NEGATIVE
                and state is Disposition.PRESENT
            )
        )
        result.append(
            ObjectBongardRubricSupportDiagnostic(
                candidate.candidate_digest,
                tuple(sorted(definite)),
                tuple(sorted(
                    panel_id
                    for panel_id, state in zip(panel_ids, row, strict=True)
                    if state is Disposition.INDETERMINATE
                )),
                tuple(sorted(
                    panel_id
                    for panel_id, state in zip(panel_ids, row, strict=True)
                    if state is Disposition.ERROR
                )),
            )
        )
    return tuple(sorted(result, key=lambda item: item.candidate_digest))


def _make_support_gap(
    candidates: Sequence[ObjectBongardRubricCandidate],
    panel_ids: Sequence[str],
    sides: Sequence[RubricSupportSide],
    rows: Sequence[Sequence[Disposition]],
) -> ObjectBongardRubricSupportGap:
    diagnostics = _support_diagnostics(candidates, panel_ids, sides, rows)
    # LANGUAGE_GAP means that every member of the complete finite language is
    # refuted by at least one definite support counterexample.  If even one
    # otherwise viable member is blocked only by missing/uncertain/error
    # evidence, the correct diagnosis is WITNESS_GAP.
    kind = (
        RubricSupportGapKind.LANGUAGE_GAP
        if all(item.definite_counterexample_panel_ids for item in diagnostics)
        else RubricSupportGapKind.WITNESS_GAP
    )
    provisional = object.__new__(ObjectBongardRubricSupportGap)
    object.__setattr__(provisional, "kind", kind)
    object.__setattr__(provisional, "diagnostics", diagnostics)
    return ObjectBongardRubricSupportGap(
        kind, diagnostics, canonical_digest(_gap_content(provisional))
    )


def _version_content(
    value: "ObjectBongardRubricSupportVersionSpace",
) -> dict[str, object]:
    return {
        "schema": RUBRIC_SUPPORT_VERSION_SPACE_SCHEMA,
        "algorithm_id": RUBRIC_VERSION_SPACE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "observer_catalog_digest": value.observer_catalog_digest,
        "observer_runtime_identity_digest": value.observer_runtime_identity_digest,
        "candidates": [item.to_data() for item in value.candidates],
        "support_panel_ids": list(value.support_panel_ids),
        "support_artifact_digests": list(value.support_artifact_digests),
        "support_sides": [item.value for item in value.support_sides],
        "rows": [[state.value for state in row] for row in value.rows],
        "survivor_candidate_digests": list(value.survivor_candidate_digests),
        "gap": None if value.gap is None else value.gap.to_data(),
        "positive_accept": Disposition.PRESENT.value,
        "negative_accept": Disposition.CERTIFIED_ABSENT.value,
        "support_panels_per_side": RUBRIC_SUPPORT_PANELS_PER_SIDE,
        "complete_finite_inventory": True,
        "codex_may_rank_survivors_only": True,
        **_language_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricSupportVersionSpace:
    """Both candidate rows and their exact support-consistent subset."""

    algorithm_digest: str
    rubric_spec_digest: str
    observer_catalog_digest: str
    observer_runtime_identity_digest: str
    candidates: tuple[ObjectBongardRubricCandidate, ...]
    support_panel_ids: tuple[str, ...]
    support_artifact_digests: tuple[str, ...]
    support_sides: tuple[RubricSupportSide, ...]
    rows: tuple[tuple[Disposition, ...], ...]
    survivor_candidate_digests: tuple[str, ...]
    gap: ObjectBongardRubricSupportGap | None
    version_space_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != (
            object_bongard_rubric_version_space_algorithm_digest()
        ):
            raise ObjectBongardRubricVersionSpaceError(
                "version-space algorithm binding differs"
            )
        for name in (
            "rubric_spec_digest",
            "observer_catalog_digest",
            "observer_runtime_identity_digest",
            "version_space_digest",
        ):
            _digest(getattr(self, name), name)
        expected_candidates = _enumerate_for_digest(self.rubric_spec_digest)
        if self.candidates != expected_candidates:
            raise ObjectBongardRubricVersionSpaceError(
                "candidate inventory is not complete and canonical"
            )
        side_size = RUBRIC_SUPPORT_PANELS_PER_SIDE
        panel_count = side_size * 2
        expected_sides = (RubricSupportSide.POSITIVE,) * side_size + (
            RubricSupportSide.NEGATIVE,
        ) * side_size
        if (
            not isinstance(self.support_panel_ids, tuple)
            or len(self.support_panel_ids) != panel_count
            or len(set(self.support_panel_ids)) != panel_count
            or any(_PANEL_ID.fullmatch(item) is None for item in self.support_panel_ids)
            or self.support_panel_ids[:side_size]
            != tuple(sorted(self.support_panel_ids[:side_size]))
            or self.support_panel_ids[side_size:]
            != tuple(sorted(self.support_panel_ids[side_size:]))
            or self.support_sides != expected_sides
        ):
            raise ObjectBongardRubricVersionSpaceError(
                "support must be exactly six sorted positive and six sorted negative panels"
            )
        if (
            not isinstance(self.support_artifact_digests, tuple)
            or len(self.support_artifact_digests) != panel_count
        ):
            raise ObjectBongardRubricVersionSpaceError(
                "support artifact digest inventory has the wrong size"
            )
        for item in self.support_artifact_digests:
            _digest(item, "support artifact digest")
        if (
            not isinstance(self.rows, tuple)
            or len(self.rows) != len(self.candidates)
            or any(
                not isinstance(row, tuple) or len(row) != panel_count
                for row in self.rows
            )
            or any(
                not isinstance(state, Disposition)
                for row in self.rows
                for state in row
            )
        ):
            raise ObjectBongardRubricVersionSpaceError(
                "version-space disposition rows are malformed"
            )
        expected_survivors = tuple(
            candidate.candidate_digest
            for candidate, row in zip(self.candidates, self.rows, strict=True)
            if _is_survivor(row, self.support_sides)
        )
        if self.survivor_candidate_digests != expected_survivors:
            raise ObjectBongardRubricVersionSpaceError(
                "survivor inventory differs from exact row replay"
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
            raise ObjectBongardRubricVersionSpaceError(
                "typed support gap differs from exact row replay"
            )
        if self.version_space_digest != canonical_digest(_version_content(self)):
            raise ObjectBongardRubricVersionSpaceError(
                "version-space digest differs from canonical content"
            )

    def row(self, candidate_digest: str) -> tuple[Disposition, ...]:
        _digest(candidate_digest, "candidate_digest")
        matches = tuple(
            row
            for candidate, row in zip(self.candidates, self.rows, strict=True)
            if candidate.candidate_digest == candidate_digest
        )
        if len(matches) != 1:
            raise ObjectBongardRubricVersionSpaceError("candidate row is absent")
        return matches[0]

    def survivor(self, candidate_digest: str) -> ObjectBongardRubricCandidate:
        _digest(candidate_digest, "candidate_digest")
        if candidate_digest not in self.survivor_candidate_digests:
            raise ObjectBongardRubricVersionSpaceError(
                "candidate is not a verified survivor"
            )
        return next(
            item for item in self.candidates if item.candidate_digest == candidate_digest
        )

    def to_data(self) -> dict[str, object]:
        return {
            **_version_content(self),
            "version_space_digest": self.version_space_digest,
        }

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardRubricSupportVersionSpace":
        raw = _fields(
            value,
            {
                "schema",
                "algorithm_id",
                "algorithm_digest",
                "rubric_spec_digest",
                "observer_catalog_digest",
                "observer_runtime_identity_digest",
                "candidates",
                "support_panel_ids",
                "support_artifact_digests",
                "support_sides",
                "rows",
                "survivor_candidate_digests",
                "gap",
                "positive_accept",
                "negative_accept",
                "support_panels_per_side",
                "complete_finite_inventory",
                "codex_may_rank_survivors_only",
                *_language_data(),
                *_authority_data(),
                "version_space_digest",
            },
            "rubric support version space",
        )
        for name in (
            "candidates",
            "support_panel_ids",
            "support_artifact_digests",
            "support_sides",
            "rows",
            "survivor_candidate_digests",
        ):
            if not isinstance(raw[name], list):
                raise ObjectBongardRubricVersionSpaceError(
                    f"{name} must be a JSON list"
                )
        try:
            sides = tuple(RubricSupportSide(item) for item in raw["support_sides"])
            rows = tuple(
                tuple(Disposition(item) for item in row) for row in raw["rows"]
            )
        except (TypeError, ValueError) as exc:
            raise ObjectBongardRubricVersionSpaceError(
                "support side or row disposition is unknown"
            ) from exc
        result = cls(
            raw["algorithm_digest"],
            raw["rubric_spec_digest"],
            raw["observer_catalog_digest"],
            raw["observer_runtime_identity_digest"],
            tuple(ObjectBongardRubricCandidate.from_data(item) for item in raw["candidates"]),
            tuple(raw["support_panel_ids"]),
            tuple(raw["support_artifact_digests"]),
            sides,
            rows,
            tuple(raw["survivor_candidate_digests"]),
            (
                None
                if raw["gap"] is None
                else ObjectBongardRubricSupportGap.from_data(raw["gap"])
            ),
            raw["version_space_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricVersionSpaceError(
                "rubric support version space is not canonical"
            )
        return result


RubricSupportVersionSpace = ObjectBongardRubricSupportVersionSpace


def _require_observation_scope(
    observation: RubricScopeObservation, scope: RubricScope
) -> None:
    if not isinstance(observation, RubricScopeObservation):
        raise TypeError("rubric observation has the wrong type")
    if observation.scope is not scope:
        raise ObjectBongardRubricVersionSpaceError(
            "rubric observation scope differs from predicate scope"
        )


def _scored_bounds(observation: RubricScopeObservation) -> tuple[int, int]:
    if observation.state is not RubricObservationState.SCORED:
        raise ObjectBongardRubricVersionSpaceError(
            "only a scored observation has ordinal bounds"
        )
    interval = observation.interval
    if interval is None:
        raise ObjectBongardRubricVersionSpaceError(
            "scored observation is missing its ordinal interval"
        )
    return interval.lower, interval.upper


def _evaluate_object_scope(
    stable: Sequence[RubricScopeObservation],
    unresolved: Sequence[RubricScopeObservation],
    threshold: int,
) -> Disposition:
    _threshold(threshold)
    stable_values = tuple(stable)
    unresolved_values = tuple(unresolved)
    for item in (*stable_values, *unresolved_values):
        _require_observation_scope(item, RubricScope.OBJECT)

    # Only an admitted stable object may establish the existential witness.
    # An unresolved possible object is deliberately one-way evidence: it may
    # prevent absence, but never upgrades itself into a positive object fact.
    if any(
        item.state is RubricObservationState.SCORED
        and _scored_bounds(item)[0] >= threshold
        for item in stable_values
    ):
        return Disposition.PRESENT

    # Absence is considerably stricter.  At least one stable object must have
    # been admitted, and every stable as well as every eligible unresolved
    # possible-object observation must certify foil preference with an upper
    # bound at most 1.  A tie at level 2 is deliberately indeterminate.
    # Arbitrary unions never reach either observer inventory.
    all_values = stable_values + unresolved_values
    if stable_values and all(
        item.state is RubricObservationState.SCORED
        and _scored_bounds(item)[1] <= RUBRIC_ABSENCE_UPPER_BOUND
        for item in all_values
    ):
        return Disposition.CERTIFIED_ABSENT

    # A transport/model/parser error dominates remaining uncertainty, but it
    # cannot erase a stable positive witness handled above.
    if any(item.state is RubricObservationState.ERROR for item in all_values):
        return Disposition.ERROR
    return Disposition.INDETERMINATE


def _evaluate_scene_scope(
    observation: RubricScopeObservation | None, threshold: int
) -> Disposition:
    _threshold(threshold)
    if observation is None:
        return Disposition.INDETERMINATE
    _require_observation_scope(observation, RubricScope.SCENE)
    if observation.state is RubricObservationState.ERROR:
        return Disposition.ERROR
    if observation.state is RubricObservationState.INDETERMINATE:
        return Disposition.INDETERMINATE
    lower, upper = _scored_bounds(observation)
    if lower >= threshold:
        return Disposition.PRESENT
    if upper <= RUBRIC_ABSENCE_UPPER_BOUND:
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def _validated_artifact(
    artifact: ObjectBongardRubricObserverArtifact,
) -> ObjectBongardRubricObserverArtifact:
    if not isinstance(artifact, ObjectBongardRubricObserverArtifact):
        raise TypeError("artifact must be ObjectBongardRubricObserverArtifact")
    return ObjectBongardRubricObserverArtifact.from_data(artifact.to_data())


def evaluate_object_bongard_rubric_candidate(
    candidate: ObjectBongardRubricCandidate,
    artifact: ObjectBongardRubricObserverArtifact,
) -> ObjectBongardRubricCandidateEvaluation:
    """Evaluate one frozen candidate without consulting a support-side label."""

    candidate = ObjectBongardRubricCandidate.from_data(candidate.to_data())
    artifact = _validated_artifact(artifact)
    if candidate.rubric_spec_digest != artifact.rubric_spec.spec_digest:
        raise ObjectBongardRubricVersionSpaceError(
            "candidate and observer rubric spec digests differ"
        )
    if candidate not in _enumerate_for_digest(candidate.rubric_spec_digest):
        raise ObjectBongardRubricVersionSpaceError(
            "candidate is outside the complete frozen inventory"
        )
    if candidate.scope is RubricScope.OBJECT:
        disposition = _evaluate_object_scope(
            artifact.object_observations,
            artifact.unresolved_object_observations,
            candidate.threshold,
        )
    else:
        disposition = _evaluate_scene_scope(
            artifact.canonical_scene_observation, candidate.threshold
        )
    values = {
        "algorithm_digest": candidate.algorithm_digest,
        "rubric_spec_digest": candidate.rubric_spec_digest,
        "candidate_digest": candidate.candidate_digest,
        "observer_artifact_digest": artifact.artifact_digest,
        "panel_id": artifact.panel_id,
        "disposition": disposition,
    }
    provisional = object.__new__(ObjectBongardRubricCandidateEvaluation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricCandidateEvaluation(
        **values,  # type: ignore[arg-type]
        evaluation_digest=canonical_digest(_evaluation_content(provisional)),
    )


def _canonical_support_artifacts(
    values: Sequence[ObjectBongardRubricObserverArtifact],
    *,
    label: str,
) -> tuple[ObjectBongardRubricObserverArtifact, ...]:
    frozen = tuple(
        sorted((_validated_artifact(item) for item in values), key=lambda item: item.panel_id)
    )
    if len(frozen) != RUBRIC_SUPPORT_PANELS_PER_SIDE:
        raise ObjectBongardRubricVersionSpaceError(
            f"{label} support must contain exactly six observer artifacts"
        )
    return frozen


def build_object_bongard_rubric_support_version_space(
    spec: ObjectBongardRubricSpec,
    positives: Sequence[ObjectBongardRubricObserverArtifact],
    negatives: Sequence[ObjectBongardRubricObserverArtifact],
) -> ObjectBongardRubricSupportVersionSpace:
    """Enumerate and filter the exact two predicates on 6 + 6 support."""

    if not isinstance(spec, ObjectBongardRubricSpec):
        raise TypeError("spec must be ObjectBongardRubricSpec")
    spec = ObjectBongardRubricSpec.from_data(spec.to_data())
    positive_artifacts = _canonical_support_artifacts(positives, label="positive")
    negative_artifacts = _canonical_support_artifacts(negatives, label="negative")
    artifacts = positive_artifacts + negative_artifacts
    if len({item.panel_id for item in artifacts}) != len(artifacts):
        raise ObjectBongardRubricVersionSpaceError(
            "support panel IDs must be globally distinct"
        )
    if any(item.rubric_spec.spec_digest != spec.spec_digest for item in artifacts):
        raise ObjectBongardRubricVersionSpaceError(
            "support observer artifacts do not share the exact rubric spec"
        )
    catalog_digests = {item.catalog_digest for item in artifacts}
    if len(catalog_digests) != 1:
        raise ObjectBongardRubricVersionSpaceError(
            "support observer artifacts do not share one catalog identity"
        )
    runtime_digests = {item.runtime_identity_digest for item in artifacts}
    if len(runtime_digests) != 1:
        raise ObjectBongardRubricVersionSpaceError(
            "support observer artifacts do not share one runtime identity"
        )
    sides = (RubricSupportSide.POSITIVE,) * RUBRIC_SUPPORT_PANELS_PER_SIDE + (
        RubricSupportSide.NEGATIVE,
    ) * RUBRIC_SUPPORT_PANELS_PER_SIDE
    candidates = enumerate_object_bongard_rubric_candidates(spec)
    rows = tuple(
        tuple(
            evaluate_object_bongard_rubric_candidate(candidate, artifact).disposition
            for artifact in artifacts
        )
        for candidate in candidates
    )
    survivors = tuple(
        candidate.candidate_digest
        for candidate, row in zip(candidates, rows, strict=True)
        if _is_survivor(row, sides)
    )
    panel_ids = tuple(item.panel_id for item in artifacts)
    gap = (
        None
        if survivors
        else _make_support_gap(candidates, panel_ids, sides, rows)
    )
    values = {
        "algorithm_digest": object_bongard_rubric_version_space_algorithm_digest(),
        "rubric_spec_digest": spec.spec_digest,
        "observer_catalog_digest": next(iter(catalog_digests)),
        "observer_runtime_identity_digest": next(iter(runtime_digests)),
        "candidates": candidates,
        "support_panel_ids": panel_ids,
        "support_artifact_digests": tuple(item.artifact_digest for item in artifacts),
        "support_sides": sides,
        "rows": rows,
        "survivor_candidate_digests": survivors,
        "gap": gap,
    }
    provisional = object.__new__(ObjectBongardRubricSupportVersionSpace)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricSupportVersionSpace(
        **values,  # type: ignore[arg-type]
        version_space_digest=canonical_digest(_version_content(provisional)),
    )


def cold_verify_object_bongard_rubric_support_version_space(
    version_space: ObjectBongardRubricSupportVersionSpace,
    spec: ObjectBongardRubricSpec,
    positives: Sequence[ObjectBongardRubricObserverArtifact],
    negatives: Sequence[ObjectBongardRubricObserverArtifact],
) -> ObjectBongardRubricSupportVersionSpace:
    """Replay enumeration and all decisions without pixels or model calls."""

    decoded = ObjectBongardRubricSupportVersionSpace.from_data(
        version_space.to_data()
    )
    replayed = build_object_bongard_rubric_support_version_space(
        spec, positives, negatives
    )
    if decoded != replayed:
        raise ObjectBongardRubricVersionSpaceError(
            "cold rubric version-space replay differs"
        )
    return decoded


__all__ = (
    "ObjectBongardRubricCandidate",
    "ObjectBongardRubricCandidateEvaluation",
    "ObjectBongardRubricSupportDiagnostic",
    "ObjectBongardRubricSupportGap",
    "ObjectBongardRubricSupportVersionSpace",
    "ObjectBongardRubricVersionSpaceError",
    "RUBRIC_ABSENCE_UPPER_BOUND",
    "RUBRIC_PRESENT_LOWER_BOUND",
    "RUBRIC_SUPPORT_PANELS_PER_SIDE",
    "RUBRIC_THRESHOLDS",
    "RUBRIC_VERSION_SPACE_ALGORITHM_ID",
    "RubricPredicateCandidate",
    "RubricPredicateOperator",
    "RubricSupportGapKind",
    "RubricSupportSide",
    "RubricSupportVersionSpace",
    "build_object_bongard_rubric_support_version_space",
    "cold_verify_object_bongard_rubric_support_version_space",
    "enumerate_object_bongard_rubric_candidates",
    "evaluate_object_bongard_rubric_candidate",
    "object_bongard_rubric_version_space_algorithm_digest",
    "object_bongard_rubric_version_space_source_digest",
)
