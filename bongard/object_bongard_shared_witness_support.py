"""Closed Python support gate for structured shared-witness predicates.

There is exactly one orientation-preserving candidate per frozen rubric spec.
The candidate survives only when six target and six foil panel observations
meet the preregistered symmetric 5/6 rule.  Every panel artifact, including
its complete entity inventory, is retained in the version-space record.

This module contains no Lean dependency, polarity repair, negation, threshold
search, retry, ordinal sum, or model-based selection.
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
from bongard.object_bongard_shared_witness import (
    ObjectBongardSharedWitnessRubricSpec,
)
from bongard.object_bongard_shared_witness_observer import (
    ObjectBongardSharedWitnessPanelArtifact,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


SHARED_WITNESS_CANDIDATE_SCHEMA = "gkm.bongard-shared-witness-candidate.v1"
SHARED_WITNESS_EVALUATION_SCHEMA = "gkm.bongard-shared-witness-evaluation.v1"
SHARED_WITNESS_SUPPORT_GAP_SCHEMA = "gkm.bongard-shared-witness-support-gap.v1"
SHARED_WITNESS_SUPPORT_VERSION_SPACE_SCHEMA = (
    "gkm.bongard-shared-witness-support-version-space.v1"
)
SHARED_WITNESS_SUPPORT_ALGORITHM_ID = (
    "bongard.shared-witness-support/fixed-symmetric-five-of-six-v1"
)
SHARED_WITNESS_SUPPORT_PANELS_PER_SIDE = 6
SHARED_WITNESS_MIN_DEFINITE_MATCHES_PER_SIDE = 5
SHARED_WITNESS_MAX_INDETERMINATE_PER_SIDE = 1

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")


class ObjectBongardSharedWitnessSupportError(ValueError):
    """A candidate, observation inventory, or replay failed closed."""


class SharedWitnessSupportSide(str, Enum):
    TARGET = "target"
    FOIL = "foil"


class SharedWitnessSupportGapKind(str, Enum):
    LANGUAGE_GAP = "language_gap"
    WITNESS_GAP = "witness_gap"
    ERROR_GAP = "error_gap"


class SharedWitnessSupportAcceptanceTier(str, Enum):
    STRICT_EXACT = "strict_exact_six_plus_six"
    BOUNDED_ABSTENTION = "bounded_abstention_five_of_six"
    REJECTED = "rejected"


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
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "threshold_tuning_allowed": False,
        "retries_allowed": False,
        "model_selects_candidate": False,
    }


def _policy_data() -> dict[str, object]:
    return {
        "support_panels_per_side": SHARED_WITNESS_SUPPORT_PANELS_PER_SIDE,
        "minimum_definite_matches_per_side": (
            SHARED_WITNESS_MIN_DEFINITE_MATCHES_PER_SIDE
        ),
        "maximum_indeterminate_per_side": (
            SHARED_WITNESS_MAX_INDETERMINATE_PER_SIDE
        ),
        "maximum_confident_contradictions_per_side": 0,
        "maximum_errors_per_side": 0,
        "target_match": Disposition.PRESENT.value,
        "target_contradiction": Disposition.CERTIFIED_ABSENT.value,
        "foil_match": Disposition.CERTIFIED_ABSENT.value,
        "foil_contradiction": Disposition.PRESENT.value,
        "candidate_orientation": "group-0-target/group-1-foil",
        "strict_exact_is_diagnostic_only": True,
        "all_entity_observations_persisted": True,
        "failed_or_indeterminate_is_absence": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardSharedWitnessSupportError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessSupportError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def _panel_id(value: object) -> str:
    if not isinstance(value, str) or _PANEL_ID.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessSupportError("panel ID is invalid")
    return value


def object_bongard_shared_witness_support_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_bongard_shared_witness_support_policy_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-shared-witness-support-policy.v1",
            "policy_id": SHARED_WITNESS_SUPPORT_ALGORITHM_ID,
            **_policy_data(),
            **_authority_data(),
        }
    )


def object_bongard_shared_witness_support_algorithm_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-shared-witness-support-algorithm.v1",
            "algorithm_id": SHARED_WITNESS_SUPPORT_ALGORITHM_ID,
            "implementation_source_sha256": (
                object_bongard_shared_witness_support_source_digest()
            ),
            "support_policy_digest": (
                object_bongard_shared_witness_support_policy_digest()
            ),
            "complete_candidate_count_per_spec": 1,
            **_authority_data(),
        }
    )


def _candidate_content(value: "ObjectBongardSharedWitnessCandidate") -> dict[str, object]:
    return {
        "schema": SHARED_WITNESS_CANDIDATE_SCHEMA,
        "algorithm_id": SHARED_WITNESS_SUPPORT_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "candidate_rank": value.candidate_rank,
        "candidate_id": "shared-witness:group-0-target",
        "formula": "shared_witness_has_group_0_axis_endpoint",
        "full_ir_owned_by_rubric_spec": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessCandidate:
    rubric_spec_digest: str
    candidate_rank: int
    algorithm_digest: str
    candidate_digest: str

    def __post_init__(self) -> None:
        _digest(self.rubric_spec_digest, "rubric spec digest")
        if type(self.candidate_rank) is not int or self.candidate_rank not in (0, 1):
            raise ObjectBongardSharedWitnessSupportError("candidate rank differs")
        if self.algorithm_digest != object_bongard_shared_witness_support_algorithm_digest():
            raise ObjectBongardSharedWitnessSupportError("candidate algorithm differs")
        _digest(self.candidate_digest, "candidate digest")
        if self.candidate_digest != canonical_digest(_candidate_content(self)):
            raise ObjectBongardSharedWitnessSupportError("candidate digest differs")

    @classmethod
    def create(
        cls, spec: ObjectBongardSharedWitnessRubricSpec
    ) -> "ObjectBongardSharedWitnessCandidate":
        if not isinstance(spec, ObjectBongardSharedWitnessRubricSpec):
            raise TypeError("spec must be ObjectBongardSharedWitnessRubricSpec")
        frozen = ObjectBongardSharedWitnessRubricSpec.from_data(spec.to_data())
        values = {
            "rubric_spec_digest": frozen.spec_digest,
            "candidate_rank": frozen.candidate_rank,
            "algorithm_digest": object_bongard_shared_witness_support_algorithm_digest(),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, candidate_digest=canonical_digest(_candidate_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_candidate_content(self), "candidate_digest": self.candidate_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessCandidate":
        raw = _fields(
            value,
            {
                "schema", "algorithm_id", "algorithm_digest", "rubric_spec_digest",
                "candidate_rank", "candidate_id", "formula",
                "full_ir_owned_by_rubric_spec", *_authority_data(), "candidate_digest",
            },
            "shared-witness candidate",
        )
        if (
            raw["schema"] != SHARED_WITNESS_CANDIDATE_SCHEMA
            or raw["algorithm_id"] != SHARED_WITNESS_SUPPORT_ALGORITHM_ID
            or raw["candidate_id"] != "shared-witness:group-0-target"
            or raw["formula"] != "shared_witness_has_group_0_axis_endpoint"
            or raw["full_ir_owned_by_rubric_spec"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardSharedWitnessSupportError("candidate policy differs")
        result = cls(
            raw["rubric_spec_digest"], raw["candidate_rank"],
            raw["algorithm_digest"], raw["candidate_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessSupportError("candidate is not canonical")
        return result


def _canonical_artifact(
    value: ObjectBongardSharedWitnessPanelArtifact,
) -> ObjectBongardSharedWitnessPanelArtifact:
    if not isinstance(value, ObjectBongardSharedWitnessPanelArtifact):
        raise TypeError("support evidence must be shared-witness panel artifacts")
    restored = ObjectBongardSharedWitnessPanelArtifact.from_data(value.to_data())
    if restored != value:
        raise ObjectBongardSharedWitnessSupportError("observer artifact round trip differs")
    return restored


def _evaluation_content(
    value: "ObjectBongardSharedWitnessCandidateEvaluation",
) -> dict[str, object]:
    return {
        "schema": SHARED_WITNESS_EVALUATION_SCHEMA,
        "algorithm_digest": value.algorithm_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "candidate_digest": value.candidate_digest,
        "observer_artifact_digest": value.observer_artifact_digest,
        "panel_id": value.panel_id,
        "disposition": value.disposition.value,
        "observation_source": "python-aggregated-all-entity-shared-witness-observation",
        "failed_or_indeterminate_is_absence": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessCandidateEvaluation:
    algorithm_digest: str
    rubric_spec_digest: str
    candidate_digest: str
    observer_artifact_digest: str
    panel_id: str
    disposition: Disposition
    evaluation_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != object_bongard_shared_witness_support_algorithm_digest():
            raise ObjectBongardSharedWitnessSupportError("evaluation algorithm differs")
        for name in ("rubric_spec_digest", "candidate_digest", "observer_artifact_digest", "evaluation_digest"):
            _digest(getattr(self, name), name)
        _panel_id(self.panel_id)
        if not isinstance(self.disposition, Disposition):
            raise TypeError("evaluation disposition must be Disposition")
        if self.evaluation_digest != canonical_digest(_evaluation_content(self)):
            raise ObjectBongardSharedWitnessSupportError("evaluation digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_evaluation_content(self), "evaluation_digest": self.evaluation_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessCandidateEvaluation":
        raw = _fields(
            value,
            {
                "schema", "algorithm_digest", "rubric_spec_digest", "candidate_digest",
                "observer_artifact_digest", "panel_id", "disposition", "observation_source",
                "failed_or_indeterminate_is_absence", *_authority_data(), "evaluation_digest",
            },
            "shared-witness evaluation",
        )
        if (
            raw["schema"] != SHARED_WITNESS_EVALUATION_SCHEMA
            or raw["observation_source"] != "python-aggregated-all-entity-shared-witness-observation"
            or raw["failed_or_indeterminate_is_absence"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardSharedWitnessSupportError("evaluation policy differs")
        try:
            disposition = Disposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardSharedWitnessSupportError("unknown disposition") from exc
        result = cls(
            raw["algorithm_digest"], raw["rubric_spec_digest"], raw["candidate_digest"],
            raw["observer_artifact_digest"], raw["panel_id"], disposition,
            raw["evaluation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessSupportError("evaluation is not canonical")
        return result


def evaluate_object_bongard_shared_witness_candidate(
    candidate: ObjectBongardSharedWitnessCandidate,
    artifact: ObjectBongardSharedWitnessPanelArtifact,
) -> ObjectBongardSharedWitnessCandidateEvaluation:
    candidate = ObjectBongardSharedWitnessCandidate.from_data(candidate.to_data())
    artifact = _canonical_artifact(artifact)
    if artifact.rubric_spec_digest != candidate.rubric_spec_digest:
        raise ObjectBongardSharedWitnessSupportError("candidate and artifact specs differ")
    values = {
        "algorithm_digest": candidate.algorithm_digest,
        "rubric_spec_digest": candidate.rubric_spec_digest,
        "candidate_digest": candidate.candidate_digest,
        "observer_artifact_digest": artifact.artifact_digest,
        "panel_id": artifact.panel_id,
        "disposition": artifact.observation.disposition,
    }
    provisional = object.__new__(ObjectBongardSharedWitnessCandidateEvaluation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardSharedWitnessCandidateEvaluation(
        **values, evaluation_digest=canonical_digest(_evaluation_content(provisional))
    )


def _gap_content(value: "ObjectBongardSharedWitnessSupportGap") -> dict[str, object]:
    return {
        "schema": SHARED_WITNESS_SUPPORT_GAP_SCHEMA,
        "kind": value.kind.value,
        "candidate_digest": value.candidate_digest,
        "contradiction_panel_ids": list(value.contradiction_panel_ids),
        "indeterminate_panel_ids": list(value.indeterminate_panel_ids),
        "error_panel_ids": list(value.error_panel_ids),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessSupportGap:
    kind: SharedWitnessSupportGapKind
    candidate_digest: str
    contradiction_panel_ids: tuple[str, ...]
    indeterminate_panel_ids: tuple[str, ...]
    error_panel_ids: tuple[str, ...]
    gap_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SharedWitnessSupportGapKind):
            raise TypeError("gap kind must be typed")
        _digest(self.candidate_digest, "gap candidate digest")
        inventories = []
        for name in ("contradiction_panel_ids", "indeterminate_panel_ids", "error_panel_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise ObjectBongardSharedWitnessSupportError("gap inventory differs")
            for item in values:
                _panel_id(item)
            inventories.append(set(values))
        if any(inventories[i] & inventories[j] for i in range(3) for j in range(i + 1, 3)):
            raise ObjectBongardSharedWitnessSupportError("gap inventories overlap")
        expected_kind = (
            SharedWitnessSupportGapKind.ERROR_GAP if self.error_panel_ids else
            SharedWitnessSupportGapKind.LANGUAGE_GAP if self.contradiction_panel_ids else
            SharedWitnessSupportGapKind.WITNESS_GAP
        )
        if self.kind is not expected_kind or self.gap_digest != canonical_digest(_gap_content(self)):
            raise ObjectBongardSharedWitnessSupportError("gap kind or digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessSupportGap":
        raw = _fields(
            value,
            {"schema", "kind", "candidate_digest", "contradiction_panel_ids", "indeterminate_panel_ids", "error_panel_ids", "gap_digest"},
            "shared-witness support gap",
        )
        if raw["schema"] != SHARED_WITNESS_SUPPORT_GAP_SCHEMA or any(
            not isinstance(raw[name], list)
            for name in ("contradiction_panel_ids", "indeterminate_panel_ids", "error_panel_ids")
        ):
            raise ObjectBongardSharedWitnessSupportError("gap policy differs")
        try:
            kind = SharedWitnessSupportGapKind(raw["kind"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardSharedWitnessSupportError("unknown gap kind") from exc
        result = cls(
            kind, raw["candidate_digest"], tuple(raw["contradiction_panel_ids"]),
            tuple(raw["indeterminate_panel_ids"]), tuple(raw["error_panel_ids"]),
            raw["gap_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessSupportError("gap is not canonical")
        return result


def _bounded_admissible(target: Sequence[Disposition], foil: Sequence[Disposition]) -> bool:
    target = tuple(target)
    foil = tuple(foil)
    if len(target) != 6 or len(foil) != 6:
        raise ObjectBongardSharedWitnessSupportError("support rule requires six panels per side")
    return (
        target.count(Disposition.PRESENT) >= 5
        and target.count(Disposition.CERTIFIED_ABSENT) == 0
        and target.count(Disposition.ERROR) == 0
        and target.count(Disposition.INDETERMINATE) <= 1
        and foil.count(Disposition.CERTIFIED_ABSENT) >= 5
        and foil.count(Disposition.PRESENT) == 0
        and foil.count(Disposition.ERROR) == 0
        and foil.count(Disposition.INDETERMINATE) <= 1
    )


def _make_gap(
    candidate: ObjectBongardSharedWitnessCandidate,
    panel_ids: Sequence[str],
    sides: Sequence[SharedWitnessSupportSide],
    row: Sequence[Disposition],
) -> ObjectBongardSharedWitnessSupportGap:
    contradictions = tuple(sorted(
        panel_id for panel_id, side, state in zip(panel_ids, sides, row, strict=True)
        if (side is SharedWitnessSupportSide.TARGET and state is Disposition.CERTIFIED_ABSENT)
        or (side is SharedWitnessSupportSide.FOIL and state is Disposition.PRESENT)
    ))
    indeterminate = tuple(sorted(
        panel_id for panel_id, state in zip(panel_ids, row, strict=True)
        if state is Disposition.INDETERMINATE
    ))
    errors = tuple(sorted(
        panel_id for panel_id, state in zip(panel_ids, row, strict=True)
        if state is Disposition.ERROR
    ))
    kind = (
        SharedWitnessSupportGapKind.ERROR_GAP if errors else
        SharedWitnessSupportGapKind.LANGUAGE_GAP if contradictions else
        SharedWitnessSupportGapKind.WITNESS_GAP
    )
    values = {
        "kind": kind,
        "candidate_digest": candidate.candidate_digest,
        "contradiction_panel_ids": contradictions,
        "indeterminate_panel_ids": indeterminate,
        "error_panel_ids": errors,
    }
    provisional = object.__new__(ObjectBongardSharedWitnessSupportGap)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardSharedWitnessSupportGap(
        **values, gap_digest=canonical_digest(_gap_content(provisional))
    )


def _version_content(value: "ObjectBongardSharedWitnessSupportVersionSpace") -> dict[str, object]:
    return {
        "schema": SHARED_WITNESS_SUPPORT_VERSION_SPACE_SCHEMA,
        "algorithm_id": SHARED_WITNESS_SUPPORT_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "support_policy_digest": object_bongard_shared_witness_support_policy_digest(),
        "rubric_spec_digest": value.rubric_spec_digest,
        "observer_protocol_digest": value.observer_protocol_digest,
        "observer_runtime_identity_digest": value.observer_runtime_identity_digest,
        "candidate": value.candidate.to_data(),
        "support_artifacts": [item.to_data() for item in value.support_artifacts],
        "support_panel_ids": list(value.support_panel_ids),
        "support_sides": [item.value for item in value.support_sides],
        "row": [item.value for item in value.row],
        "survivor_candidate_digests": list(value.survivor_candidate_digests),
        "strict_survivor_candidate_digests": list(value.strict_survivor_candidate_digests),
        "support_acceptance_tier": value.support_acceptance_tier.value,
        "gap": None if value.gap is None else value.gap.to_data(),
        "complete_entity_observations_embedded": True,
        **_policy_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessSupportVersionSpace:
    algorithm_digest: str
    rubric_spec_digest: str
    observer_protocol_digest: str
    observer_runtime_identity_digest: str
    candidate: ObjectBongardSharedWitnessCandidate
    support_artifacts: tuple[ObjectBongardSharedWitnessPanelArtifact, ...]
    support_panel_ids: tuple[str, ...]
    support_sides: tuple[SharedWitnessSupportSide, ...]
    row: tuple[Disposition, ...]
    survivor_candidate_digests: tuple[str, ...]
    strict_survivor_candidate_digests: tuple[str, ...]
    support_acceptance_tier: SharedWitnessSupportAcceptanceTier
    gap: ObjectBongardSharedWitnessSupportGap | None
    version_space_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != object_bongard_shared_witness_support_algorithm_digest():
            raise ObjectBongardSharedWitnessSupportError("version-space algorithm differs")
        for name in ("rubric_spec_digest", "observer_protocol_digest", "observer_runtime_identity_digest", "version_space_digest"):
            _digest(getattr(self, name), name)
        if self.candidate.rubric_spec_digest != self.rubric_spec_digest:
            raise ObjectBongardSharedWitnessSupportError("version-space candidate differs")
        expected_sides = (SharedWitnessSupportSide.TARGET,) * 6 + (SharedWitnessSupportSide.FOIL,) * 6
        if (
            not isinstance(self.support_artifacts, tuple) or len(self.support_artifacts) != 12
            or self.support_panel_ids != tuple(item.panel_id for item in self.support_artifacts)
            or len(set(self.support_panel_ids)) != 12
            or self.support_panel_ids[:6] != tuple(sorted(self.support_panel_ids[:6]))
            or self.support_panel_ids[6:] != tuple(sorted(self.support_panel_ids[6:]))
            or self.support_sides != expected_sides
            or self.row != tuple(item.observation.disposition for item in self.support_artifacts)
            or any(item.rubric_spec_digest != self.rubric_spec_digest for item in self.support_artifacts)
            or any(item.protocol_digest != self.observer_protocol_digest for item in self.support_artifacts)
            or any(item.runtime_identity_digest != self.observer_runtime_identity_digest for item in self.support_artifacts)
        ):
            raise ObjectBongardSharedWitnessSupportError("support inventory differs")
        strict = self.row[:6] == (Disposition.PRESENT,) * 6 and self.row[6:] == (Disposition.CERTIFIED_ABSENT,) * 6
        survivor = _bounded_admissible(self.row[:6], self.row[6:])
        expected_survivors = (self.candidate.candidate_digest,) if survivor else ()
        expected_strict = (self.candidate.candidate_digest,) if strict else ()
        expected_tier = (
            SharedWitnessSupportAcceptanceTier.STRICT_EXACT if strict else
            SharedWitnessSupportAcceptanceTier.BOUNDED_ABSTENTION if survivor else
            SharedWitnessSupportAcceptanceTier.REJECTED
        )
        expected_gap = None if survivor else _make_gap(self.candidate, self.support_panel_ids, self.support_sides, self.row)
        if (
            self.survivor_candidate_digests != expected_survivors
            or self.strict_survivor_candidate_digests != expected_strict
            or self.support_acceptance_tier is not expected_tier
            or self.gap != expected_gap
            or self.version_space_digest != canonical_digest(_version_content(self))
        ):
            raise ObjectBongardSharedWitnessSupportError("support decision or digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_version_content(self), "version_space_digest": self.version_space_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessSupportVersionSpace":
        raw = _fields(
            value,
            {
                "schema", "algorithm_id", "algorithm_digest", "support_policy_digest",
                "rubric_spec_digest", "observer_protocol_digest", "observer_runtime_identity_digest",
                "candidate", "support_artifacts", "support_panel_ids", "support_sides", "row",
                "survivor_candidate_digests", "strict_survivor_candidate_digests",
                "support_acceptance_tier", "gap", "complete_entity_observations_embedded",
                *_policy_data(), *_authority_data(), "version_space_digest",
            },
            "shared-witness support version space",
        )
        if (
            raw["schema"] != SHARED_WITNESS_SUPPORT_VERSION_SPACE_SCHEMA
            or raw["algorithm_id"] != SHARED_WITNESS_SUPPORT_ALGORITHM_ID
            or raw["support_policy_digest"] != object_bongard_shared_witness_support_policy_digest()
            or raw["complete_entity_observations_embedded"] is not True
            or any(raw[key] != item for key, item in _policy_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(not isinstance(raw[name], list) for name in (
                "support_artifacts", "support_panel_ids", "support_sides", "row",
                "survivor_candidate_digests", "strict_survivor_candidate_digests",
            ))
        ):
            raise ObjectBongardSharedWitnessSupportError("version-space policy differs")
        try:
            result = cls(
                raw["algorithm_digest"], raw["rubric_spec_digest"], raw["observer_protocol_digest"],
                raw["observer_runtime_identity_digest"],
                ObjectBongardSharedWitnessCandidate.from_data(raw["candidate"]),
                tuple(ObjectBongardSharedWitnessPanelArtifact.from_data(item) for item in raw["support_artifacts"]),
                tuple(raw["support_panel_ids"]),
                tuple(SharedWitnessSupportSide(item) for item in raw["support_sides"]),
                tuple(Disposition(item) for item in raw["row"]),
                tuple(raw["survivor_candidate_digests"]),
                tuple(raw["strict_survivor_candidate_digests"]),
                SharedWitnessSupportAcceptanceTier(raw["support_acceptance_tier"]),
                None if raw["gap"] is None else ObjectBongardSharedWitnessSupportGap.from_data(raw["gap"]),
                raw["version_space_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise ObjectBongardSharedWitnessSupportError("version space is malformed") from exc
        if result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessSupportError("version space is not canonical")
        return result


def _canonical_side(
    values: Sequence[ObjectBongardSharedWitnessPanelArtifact], label: str
) -> tuple[ObjectBongardSharedWitnessPanelArtifact, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} support must be a sequence")
    frozen = tuple(sorted((_canonical_artifact(item) for item in values), key=lambda item: item.panel_id))
    if len(frozen) != 6:
        raise ObjectBongardSharedWitnessSupportError(f"{label} support must contain exactly six artifacts")
    return frozen


def build_object_bongard_shared_witness_support_version_space(
    spec: ObjectBongardSharedWitnessRubricSpec,
    targets: Sequence[ObjectBongardSharedWitnessPanelArtifact],
    foils: Sequence[ObjectBongardSharedWitnessPanelArtifact],
) -> ObjectBongardSharedWitnessSupportVersionSpace:
    if not isinstance(spec, ObjectBongardSharedWitnessRubricSpec):
        raise TypeError("spec must be ObjectBongardSharedWitnessRubricSpec")
    spec = ObjectBongardSharedWitnessRubricSpec.from_data(spec.to_data())
    target_artifacts = _canonical_side(targets, "target")
    foil_artifacts = _canonical_side(foils, "foil")
    artifacts = target_artifacts + foil_artifacts
    if len({item.panel_id for item in artifacts}) != 12:
        raise ObjectBongardSharedWitnessSupportError("support panel IDs must be distinct")
    if any(item.rubric_spec_digest != spec.spec_digest for item in artifacts):
        raise ObjectBongardSharedWitnessSupportError("support spec binding differs")
    protocols = {item.protocol_digest for item in artifacts}
    runtimes = {item.runtime_identity_digest for item in artifacts}
    if len(protocols) != 1 or len(runtimes) != 1:
        raise ObjectBongardSharedWitnessSupportError("support protocol or runtime differs")
    candidate = ObjectBongardSharedWitnessCandidate.create(spec)
    row = tuple(item.observation.disposition for item in artifacts)
    sides = (SharedWitnessSupportSide.TARGET,) * 6 + (SharedWitnessSupportSide.FOIL,) * 6
    survivor = _bounded_admissible(row[:6], row[6:])
    strict = row[:6] == (Disposition.PRESENT,) * 6 and row[6:] == (Disposition.CERTIFIED_ABSENT,) * 6
    values = {
        "algorithm_digest": object_bongard_shared_witness_support_algorithm_digest(),
        "rubric_spec_digest": spec.spec_digest,
        "observer_protocol_digest": next(iter(protocols)),
        "observer_runtime_identity_digest": next(iter(runtimes)),
        "candidate": candidate,
        "support_artifacts": artifacts,
        "support_panel_ids": tuple(item.panel_id for item in artifacts),
        "support_sides": sides,
        "row": row,
        "survivor_candidate_digests": (candidate.candidate_digest,) if survivor else (),
        "strict_survivor_candidate_digests": (candidate.candidate_digest,) if strict else (),
        "support_acceptance_tier": (
            SharedWitnessSupportAcceptanceTier.STRICT_EXACT if strict else
            SharedWitnessSupportAcceptanceTier.BOUNDED_ABSTENTION if survivor else
            SharedWitnessSupportAcceptanceTier.REJECTED
        ),
        "gap": None if survivor else _make_gap(candidate, tuple(item.panel_id for item in artifacts), sides, row),
    }
    provisional = object.__new__(ObjectBongardSharedWitnessSupportVersionSpace)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardSharedWitnessSupportVersionSpace(
        **values, version_space_digest=canonical_digest(_version_content(provisional))
    )


def cold_verify_object_bongard_shared_witness_support_version_space(
    version_space: ObjectBongardSharedWitnessSupportVersionSpace,
    spec: ObjectBongardSharedWitnessRubricSpec,
    targets: Sequence[ObjectBongardSharedWitnessPanelArtifact],
    foils: Sequence[ObjectBongardSharedWitnessPanelArtifact],
) -> ObjectBongardSharedWitnessSupportVersionSpace:
    decoded = ObjectBongardSharedWitnessSupportVersionSpace.from_data(version_space.to_data())
    replayed = build_object_bongard_shared_witness_support_version_space(spec, targets, foils)
    if decoded != replayed:
        raise ObjectBongardSharedWitnessSupportError("cold support replay differs")
    return decoded


__all__ = (
    "ObjectBongardSharedWitnessCandidate",
    "ObjectBongardSharedWitnessCandidateEvaluation",
    "ObjectBongardSharedWitnessSupportError",
    "ObjectBongardSharedWitnessSupportGap",
    "ObjectBongardSharedWitnessSupportVersionSpace",
    "SHARED_WITNESS_MAX_INDETERMINATE_PER_SIDE",
    "SHARED_WITNESS_MIN_DEFINITE_MATCHES_PER_SIDE",
    "SHARED_WITNESS_SUPPORT_ALGORITHM_ID",
    "SHARED_WITNESS_SUPPORT_PANELS_PER_SIDE",
    "SharedWitnessSupportAcceptanceTier",
    "SharedWitnessSupportGapKind",
    "SharedWitnessSupportSide",
    "build_object_bongard_shared_witness_support_version_space",
    "cold_verify_object_bongard_shared_witness_support_version_space",
    "evaluate_object_bongard_shared_witness_candidate",
    "object_bongard_shared_witness_support_algorithm_digest",
    "object_bongard_shared_witness_support_policy_digest",
    "object_bongard_shared_witness_support_source_digest",
)
