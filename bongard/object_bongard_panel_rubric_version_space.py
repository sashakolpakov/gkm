"""Closed Python version space for one whole-panel soft-rubric predicate.

For one frozen prose contrast there is exactly one executable candidate:
the complete panel's target-preference interval has lower bound at least
three.  The vision model supplies the interval; this module alone maps the
already sealed observation to one of four dispositions and filters the
candidate on six positive plus six negative support panels.

There is no object/scene scope choice, negation, polarity repair, threshold
search, disjunction, arbitrary code, model ranking, or Lean dependency.
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
from bongard.object_bongard_panel_rubric_observer import (
    ObjectBongardPanelRubricArtifact,
    object_bongard_panel_rubric_protocol_digest,
)
from bongard.object_bongard_rubric_language import (
    ObjectBongardRubricSpec,
    RUBRIC_ABSENCE_UPPER_BOUND,
    RUBRIC_PRESENT_LOWER_BOUND,
    object_bongard_rubric_language_source_digest,
    object_bongard_rubric_ordinal_scale_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


PANEL_RUBRIC_CANDIDATE_SCHEMA = "gkm.bongard-panel-rubric-candidate.v1"
PANEL_RUBRIC_EVALUATION_SCHEMA = "gkm.bongard-panel-rubric-evaluation.v1"
PANEL_RUBRIC_SUPPORT_DIAGNOSTIC_SCHEMA = (
    "gkm.bongard-panel-rubric-support-diagnostic.v1"
)
PANEL_RUBRIC_SUPPORT_GAP_SCHEMA = "gkm.bongard-panel-rubric-support-gap.v1"
PANEL_RUBRIC_VERSION_SPACE_SCHEMA = (
    "gkm.bongard-panel-rubric-support-version-space.v1"
)
PANEL_RUBRIC_VERSION_SPACE_ALGORITHM_ID = (
    "bongard.panel-rubric-version-space/single-panel-prefers-target-v1"
)
PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE = 6
PANEL_RUBRIC_CANDIDATE_COUNT = 1
PANEL_RUBRIC_MIN_DEFINITE_MATCHES_PER_SIDE = 5
PANEL_RUBRIC_MAX_INDETERMINATE_PER_SIDE = 1

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PANEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}\Z")


class ObjectBongardPanelRubricVersionSpaceError(ValueError):
    """A panel candidate, evaluation, or support inventory is malformed."""


class PanelRubricPredicateOperator(str, Enum):
    AT_LEAST = "at_least"


class PanelRubricSupportSide(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class PanelRubricSupportGapKind(str, Enum):
    LANGUAGE_GAP = "language_gap"
    WITNESS_GAP = "witness_gap"


class PanelRubricSupportAcceptanceTier(str, Enum):
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
        "model_selects_candidate": False,
    }


def _language_data() -> dict[str, object]:
    return {
        "scope": "panel",
        "operator": PanelRubricPredicateOperator.AT_LEAST.value,
        "threshold": RUBRIC_PRESENT_LOWER_BOUND,
        "formula": (
            f"PANEL target_preference_level >= {RUBRIC_PRESENT_LOWER_BOUND}"
        ),
        "present_when_interval_lower_at_least": RUBRIC_PRESENT_LOWER_BOUND,
        "certified_absent_when_interval_upper_at_most": (
            RUBRIC_ABSENCE_UPPER_BOUND
        ),
        "certified_absent_observation_meaning": "foil_preferred",
        "certified_absent_predicate_meaning": (
            "signed-target-preference-predicate-false"
        ),
        "literal_absence_of_visual_cue_claimed": False,
        "deadband_levels": [2],
        "tie_can_certify_absence": False,
        "complete_candidate_count_per_spec": PANEL_RUBRIC_CANDIDATE_COUNT,
        "rubric_language_source_digest": (
            object_bongard_rubric_language_source_digest()
        ),
        "ordinal_scale_digest": object_bongard_rubric_ordinal_scale_digest(),
        "observer_protocol_digest": object_bongard_panel_rubric_protocol_digest(),
        "failed_indeterminate_or_error_is_absence": False,
        "arbitrary_code_allowed": False,
    }


def object_bongard_panel_rubric_support_policy_digest() -> str:
    """Content address the preregistered no-contradiction support policy."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-rubric-support-policy.v1",
            "policy_id": (
                "bongard.panel-rubric-support/"
                "bounded-abstention-no-contradiction-v1"
            ),
            "panels_per_side": PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE,
            "minimum_definite_matches_per_side": (
                PANEL_RUBRIC_MIN_DEFINITE_MATCHES_PER_SIDE
            ),
            "maximum_indeterminate_per_side": (
                PANEL_RUBRIC_MAX_INDETERMINATE_PER_SIDE
            ),
            "maximum_confident_contradictions_per_side": 0,
            "maximum_errors_per_side": 0,
            "target_side_match": Disposition.PRESENT.value,
            "target_side_contradiction": Disposition.CERTIFIED_ABSENT.value,
            "foil_side_match": Disposition.CERTIFIED_ABSENT.value,
            "foil_side_contradiction": Disposition.PRESENT.value,
            "ordinal_sums_used": False,
            "strict_exact_six_plus_six_persisted_diagnostically": True,
            "strict_exact_required_for_admission": False,
            "all_twelve_observed_before_labels_or_selection": True,
            **_authority_data(),
        }
    )


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardPanelRubricVersionSpaceError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardPanelRubricVersionSpaceError(
            f"{label} must be a lowercase raw SHA-256"
        )
    return value


def _panel_id(value: object, label: str = "panel ID") -> str:
    if not isinstance(value, str) or _PANEL_ID.fullmatch(value) is None:
        raise ObjectBongardPanelRubricVersionSpaceError(f"{label} is invalid")
    return value


def object_bongard_panel_rubric_version_space_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_bongard_panel_rubric_version_space_algorithm_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-rubric-version-space-algorithm.v1",
            "algorithm_id": PANEL_RUBRIC_VERSION_SPACE_ALGORITHM_ID,
            "implementation_source_sha256": (
                object_bongard_panel_rubric_version_space_source_digest()
            ),
            "language": _language_data(),
            "support_panels_per_side": PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE,
            "support_policy_digest": (
                object_bongard_panel_rubric_support_policy_digest()
            ),
            "positive_accept": Disposition.PRESENT.value,
            "negative_accept": Disposition.CERTIFIED_ABSENT.value,
            **_authority_data(),
        }
    )


def _candidate_content(value: "ObjectBongardPanelRubricCandidate") -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_CANDIDATE_SCHEMA,
        "algorithm_id": PANEL_RUBRIC_VERSION_SPACE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "candidate_id": value.candidate_id,
        **_language_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricCandidate:
    """The unique PANEL >= 3 candidate for one exact rubric spec."""

    rubric_spec_digest: str
    algorithm_digest: str
    candidate_digest: str

    def __post_init__(self) -> None:
        _digest(self.rubric_spec_digest, "rubric spec digest")
        if self.algorithm_digest != object_bongard_panel_rubric_version_space_algorithm_digest():
            raise ObjectBongardPanelRubricVersionSpaceError(
                "candidate algorithm binding differs"
            )
        _digest(self.candidate_digest, "candidate digest")
        if self.candidate_digest != canonical_digest(_candidate_content(self)):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "candidate digest differs"
            )

    @property
    def candidate_id(self) -> str:
        return f"panel:at_least:{RUBRIC_PRESENT_LOWER_BOUND}"

    @property
    def scope(self) -> str:
        return "panel"

    @property
    def operator(self) -> PanelRubricPredicateOperator:
        return PanelRubricPredicateOperator.AT_LEAST

    @property
    def threshold(self) -> int:
        return RUBRIC_PRESENT_LOWER_BOUND

    @property
    def formula(self) -> str:
        return _language_data()["formula"]  # type: ignore[return-value]

    @classmethod
    def create(cls, rubric_spec_digest: str) -> "ObjectBongardPanelRubricCandidate":
        algorithm = object_bongard_panel_rubric_version_space_algorithm_digest()
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "rubric_spec_digest", rubric_spec_digest)
        object.__setattr__(provisional, "algorithm_digest", algorithm)
        return cls(
            rubric_spec_digest,
            algorithm,
            canonical_digest(_candidate_content(provisional)),
        )

    def to_data(self) -> dict[str, object]:
        return {**_candidate_content(self), "candidate_digest": self.candidate_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardPanelRubricCandidate":
        raw = _fields(
            value,
            {
                "schema", "algorithm_id", "algorithm_digest",
                "rubric_spec_digest", "candidate_id", *_language_data(),
                *_authority_data(), "candidate_digest",
            },
            "panel rubric candidate",
        )
        if (
            raw["schema"] != PANEL_RUBRIC_CANDIDATE_SCHEMA
            or raw["algorithm_id"] != PANEL_RUBRIC_VERSION_SPACE_ALGORITHM_ID
            or any(raw[key] != item for key, item in _language_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "candidate policy differs"
            )
        result = cls(
            raw["rubric_spec_digest"],
            raw["algorithm_digest"],
            raw["candidate_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "candidate is not canonical"
            )
        return result


def enumerate_object_bongard_panel_rubric_candidates(
    spec: ObjectBongardRubricSpec,
) -> tuple[ObjectBongardPanelRubricCandidate]:
    if not isinstance(spec, ObjectBongardRubricSpec):
        raise TypeError("spec must be ObjectBongardRubricSpec")
    frozen = ObjectBongardRubricSpec.from_data(spec.to_data())
    return (ObjectBongardPanelRubricCandidate.create(frozen.spec_digest),)


def _evaluation_content(
    value: "ObjectBongardPanelRubricCandidateEvaluation",
) -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_EVALUATION_SCHEMA,
        "algorithm_id": PANEL_RUBRIC_VERSION_SPACE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "candidate_digest": value.candidate_digest,
        "observer_artifact_digest": value.observer_artifact_digest,
        "panel_id": value.panel_id,
        "disposition": value.disposition.value,
        "observation_source": "one-complete-panel-ordinal-interval",
        "failed_or_missing_observation_is_absence": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricCandidateEvaluation:
    algorithm_digest: str
    rubric_spec_digest: str
    candidate_digest: str
    observer_artifact_digest: str
    panel_id: str
    disposition: Disposition
    evaluation_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != object_bongard_panel_rubric_version_space_algorithm_digest():
            raise ObjectBongardPanelRubricVersionSpaceError(
                "evaluation algorithm binding differs"
            )
        for name in (
            "rubric_spec_digest", "candidate_digest",
            "observer_artifact_digest", "evaluation_digest",
        ):
            _digest(getattr(self, name), name)
        _panel_id(self.panel_id)
        if not isinstance(self.disposition, Disposition):
            raise TypeError("evaluation disposition must be Disposition")
        expected_candidate = ObjectBongardPanelRubricCandidate.create(
            self.rubric_spec_digest
        )
        if self.candidate_digest != expected_candidate.candidate_digest:
            raise ObjectBongardPanelRubricVersionSpaceError(
                "evaluation candidate differs from singleton inventory"
            )
        if self.evaluation_digest != canonical_digest(_evaluation_content(self)):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "evaluation digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_evaluation_content(self), "evaluation_digest": self.evaluation_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardPanelRubricCandidateEvaluation":
        raw = _fields(
            value,
            {
                "schema", "algorithm_id", "algorithm_digest",
                "rubric_spec_digest", "candidate_digest",
                "observer_artifact_digest", "panel_id", "disposition",
                "observation_source", "failed_or_missing_observation_is_absence",
                *_authority_data(), "evaluation_digest",
            },
            "panel rubric evaluation",
        )
        if (
            raw["schema"] != PANEL_RUBRIC_EVALUATION_SCHEMA
            or raw["algorithm_id"] != PANEL_RUBRIC_VERSION_SPACE_ALGORITHM_ID
            or raw["observation_source"] != "one-complete-panel-ordinal-interval"
            or raw["failed_or_missing_observation_is_absence"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "evaluation policy differs"
            )
        try:
            disposition = Disposition(raw["disposition"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardPanelRubricVersionSpaceError(
                "evaluation disposition is unknown"
            ) from exc
        result = cls(
            raw["algorithm_digest"], raw["rubric_spec_digest"],
            raw["candidate_digest"], raw["observer_artifact_digest"],
            raw["panel_id"], disposition, raw["evaluation_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "evaluation is not canonical"
            )
        return result


def _validated_artifact(
    value: ObjectBongardPanelRubricArtifact,
) -> ObjectBongardPanelRubricArtifact:
    if not isinstance(value, ObjectBongardPanelRubricArtifact):
        raise TypeError("artifact must be ObjectBongardPanelRubricArtifact")
    restored = ObjectBongardPanelRubricArtifact.from_data(value.to_data())
    if restored != value:
        raise ObjectBongardPanelRubricVersionSpaceError(
            "panel observer artifact round trip differs"
        )
    return restored


def evaluate_object_bongard_panel_rubric_candidate(
    candidate: ObjectBongardPanelRubricCandidate,
    artifact: ObjectBongardPanelRubricArtifact,
) -> ObjectBongardPanelRubricCandidateEvaluation:
    candidate = ObjectBongardPanelRubricCandidate.from_data(candidate.to_data())
    artifact = _validated_artifact(artifact)
    if candidate.rubric_spec_digest != artifact.rubric_spec.spec_digest:
        raise ObjectBongardPanelRubricVersionSpaceError(
            "candidate and artifact rubric specs differ"
        )
    values = {
        "algorithm_digest": candidate.algorithm_digest,
        "rubric_spec_digest": candidate.rubric_spec_digest,
        "candidate_digest": candidate.candidate_digest,
        "observer_artifact_digest": artifact.artifact_digest,
        "panel_id": artifact.panel_id,
        "disposition": artifact.observation.disposition,
    }
    provisional = object.__new__(ObjectBongardPanelRubricCandidateEvaluation)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricCandidateEvaluation(
        **values,  # type: ignore[arg-type]
        evaluation_digest=canonical_digest(_evaluation_content(provisional)),
    )


def _diagnostic_content(
    value: "ObjectBongardPanelRubricSupportDiagnostic",
) -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_SUPPORT_DIAGNOSTIC_SCHEMA,
        "candidate_digest": value.candidate_digest,
        "definite_counterexample_panel_ids": list(
            value.definite_counterexample_panel_ids
        ),
        "indeterminate_panel_ids": list(value.indeterminate_panel_ids),
        "error_panel_ids": list(value.error_panel_ids),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricSupportDiagnostic:
    candidate_digest: str
    definite_counterexample_panel_ids: tuple[str, ...]
    indeterminate_panel_ids: tuple[str, ...]
    error_panel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _digest(self.candidate_digest, "diagnostic candidate digest")
        inventories: list[set[str]] = []
        for name in (
            "definite_counterexample_panel_ids",
            "indeterminate_panel_ids",
            "error_panel_ids",
        ):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or values != tuple(sorted(set(values)))
                or any(_PANEL_ID.fullmatch(item) is None for item in values)
            ):
                raise ObjectBongardPanelRubricVersionSpaceError(
                    f"{name} is not canonical"
                )
            inventories.append(set(values))
        if any(
            inventories[left] & inventories[right]
            for left in range(len(inventories))
            for right in range(left + 1, len(inventories))
        ):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "diagnostic inventories overlap"
            )

    def to_data(self) -> dict[str, object]:
        return _diagnostic_content(self)

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardPanelRubricSupportDiagnostic":
        raw = _fields(
            value,
            {
                "schema", "candidate_digest",
                "definite_counterexample_panel_ids", "indeterminate_panel_ids",
                "error_panel_ids",
            },
            "panel rubric support diagnostic",
        )
        if raw["schema"] != PANEL_RUBRIC_SUPPORT_DIAGNOSTIC_SCHEMA or any(
            not isinstance(raw[name], list)
            for name in (
                "definite_counterexample_panel_ids",
                "indeterminate_panel_ids", "error_panel_ids",
            )
        ):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "support diagnostic policy differs"
            )
        result = cls(
            raw["candidate_digest"],
            tuple(raw["definite_counterexample_panel_ids"]),
            tuple(raw["indeterminate_panel_ids"]),
            tuple(raw["error_panel_ids"]),
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "support diagnostic is not canonical"
            )
        return result


def _gap_content(value: "ObjectBongardPanelRubricSupportGap") -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_SUPPORT_GAP_SCHEMA,
        "kind": value.kind.value,
        "diagnostic": value.diagnostic.to_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricSupportGap:
    kind: PanelRubricSupportGapKind
    diagnostic: ObjectBongardPanelRubricSupportDiagnostic
    gap_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PanelRubricSupportGapKind):
            raise TypeError("gap kind has the wrong type")
        if not isinstance(
            self.diagnostic, ObjectBongardPanelRubricSupportDiagnostic
        ):
            raise TypeError("gap diagnostic has the wrong type")
        expected = (
            PanelRubricSupportGapKind.LANGUAGE_GAP
            if self.diagnostic.definite_counterexample_panel_ids
            else PanelRubricSupportGapKind.WITNESS_GAP
        )
        if self.kind is not expected:
            raise ObjectBongardPanelRubricVersionSpaceError(
                "gap kind differs from its support evidence"
            )
        _digest(self.gap_digest, "gap digest")
        if self.gap_digest != canonical_digest(_gap_content(self)):
            raise ObjectBongardPanelRubricVersionSpaceError("gap digest differs")

    def to_data(self) -> dict[str, object]:
        return {**_gap_content(self), "gap_digest": self.gap_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardPanelRubricSupportGap":
        raw = _fields(
            value,
            {"schema", "kind", "diagnostic", *_authority_data(), "gap_digest"},
            "panel rubric support gap",
        )
        if (
            raw["schema"] != PANEL_RUBRIC_SUPPORT_GAP_SCHEMA
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardPanelRubricVersionSpaceError("gap policy differs")
        try:
            kind = PanelRubricSupportGapKind(raw["kind"])
        except (TypeError, ValueError) as exc:
            raise ObjectBongardPanelRubricVersionSpaceError(
                "gap kind is unknown"
            ) from exc
        result = cls(
            kind,
            ObjectBongardPanelRubricSupportDiagnostic.from_data(raw["diagnostic"]),
            raw["gap_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricVersionSpaceError("gap is not canonical")
        return result


def _support_diagnostic(
    candidate: ObjectBongardPanelRubricCandidate,
    panel_ids: Sequence[str],
    sides: Sequence[PanelRubricSupportSide],
    row: Sequence[Disposition],
) -> ObjectBongardPanelRubricSupportDiagnostic:
    definite = tuple(
        sorted(
            panel_id
            for panel_id, side, state in zip(panel_ids, sides, row, strict=True)
            if (
                side is PanelRubricSupportSide.POSITIVE
                and state is Disposition.CERTIFIED_ABSENT
            )
            or (
                side is PanelRubricSupportSide.NEGATIVE
                and state is Disposition.PRESENT
            )
        )
    )
    return ObjectBongardPanelRubricSupportDiagnostic(
        candidate.candidate_digest,
        definite,
        tuple(
            sorted(
                panel_id
                for panel_id, state in zip(panel_ids, row, strict=True)
                if state is Disposition.INDETERMINATE
            )
        ),
        tuple(
            sorted(
                panel_id
                for panel_id, state in zip(panel_ids, row, strict=True)
                if state is Disposition.ERROR
            )
        ),
    )


def _make_gap(
    candidate: ObjectBongardPanelRubricCandidate,
    panel_ids: Sequence[str],
    sides: Sequence[PanelRubricSupportSide],
    row: Sequence[Disposition],
) -> ObjectBongardPanelRubricSupportGap:
    diagnostic = _support_diagnostic(candidate, panel_ids, sides, row)
    kind = (
        PanelRubricSupportGapKind.LANGUAGE_GAP
        if diagnostic.definite_counterexample_panel_ids
        else PanelRubricSupportGapKind.WITNESS_GAP
    )
    provisional = object.__new__(ObjectBongardPanelRubricSupportGap)
    object.__setattr__(provisional, "kind", kind)
    object.__setattr__(provisional, "diagnostic", diagnostic)
    return ObjectBongardPanelRubricSupportGap(
        kind, diagnostic, canonical_digest(_gap_content(provisional))
    )


def _version_content(
    value: "ObjectBongardPanelRubricSupportVersionSpace",
) -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_VERSION_SPACE_SCHEMA,
        "algorithm_id": PANEL_RUBRIC_VERSION_SPACE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "rubric_spec_digest": value.rubric_spec_digest,
        "observer_protocol_digest": value.observer_protocol_digest,
        "observer_runtime_identity_digest": value.observer_runtime_identity_digest,
        "candidate": value.candidate.to_data(),
        "support_panel_ids": list(value.support_panel_ids),
        "support_artifact_digests": list(value.support_artifact_digests),
        "support_sides": [item.value for item in value.support_sides],
        "row": [item.value for item in value.row],
        "survivor_candidate_digests": list(value.survivor_candidate_digests),
        "strict_survivor_candidate_digests": list(
            value.strict_survivor_candidate_digests
        ),
        "support_acceptance_tier": value.support_acceptance_tier.value,
        "gap": None if value.gap is None else value.gap.to_data(),
        "support_panels_per_side": PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE,
        "complete_finite_inventory": True,
        "positive_accept": Disposition.PRESENT.value,
        "negative_accept": Disposition.CERTIFIED_ABSENT.value,
        "support_policy_digest": object_bongard_panel_rubric_support_policy_digest(),
        **_language_data(),
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricSupportVersionSpace:
    algorithm_digest: str
    rubric_spec_digest: str
    observer_protocol_digest: str
    observer_runtime_identity_digest: str
    candidate: ObjectBongardPanelRubricCandidate
    support_panel_ids: tuple[str, ...]
    support_artifact_digests: tuple[str, ...]
    support_sides: tuple[PanelRubricSupportSide, ...]
    row: tuple[Disposition, ...]
    survivor_candidate_digests: tuple[str, ...]
    strict_survivor_candidate_digests: tuple[str, ...]
    support_acceptance_tier: PanelRubricSupportAcceptanceTier
    gap: ObjectBongardPanelRubricSupportGap | None
    version_space_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != object_bongard_panel_rubric_version_space_algorithm_digest():
            raise ObjectBongardPanelRubricVersionSpaceError(
                "version-space algorithm binding differs"
            )
        for name in (
            "rubric_spec_digest", "observer_protocol_digest",
            "observer_runtime_identity_digest", "version_space_digest",
        ):
            _digest(getattr(self, name), name)
        expected_candidate = ObjectBongardPanelRubricCandidate.create(
            self.rubric_spec_digest
        )
        if self.candidate != expected_candidate:
            raise ObjectBongardPanelRubricVersionSpaceError(
                "version-space candidate differs"
            )
        side_size = PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE
        expected_sides = (PanelRubricSupportSide.POSITIVE,) * side_size + (
            PanelRubricSupportSide.NEGATIVE,
        ) * side_size
        if (
            not isinstance(self.support_panel_ids, tuple)
            or len(self.support_panel_ids) != 2 * side_size
            or len(set(self.support_panel_ids)) != 2 * side_size
            or self.support_panel_ids[:side_size]
            != tuple(sorted(self.support_panel_ids[:side_size]))
            or self.support_panel_ids[side_size:]
            != tuple(sorted(self.support_panel_ids[side_size:]))
            or any(_PANEL_ID.fullmatch(item) is None for item in self.support_panel_ids)
            or self.support_sides != expected_sides
            or not isinstance(self.support_artifact_digests, tuple)
            or len(self.support_artifact_digests) != 2 * side_size
            or not isinstance(self.row, tuple)
            or len(self.row) != 2 * side_size
            or any(not isinstance(item, Disposition) for item in self.row)
        ):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "support inventory differs from exact sorted six-plus-six"
            )
        for item in self.support_artifact_digests:
            _digest(item, "support artifact digest")
        positive_row = self.row[:side_size]
        negative_row = self.row[side_size:]
        strict_survivor = (
            positive_row == (Disposition.PRESENT,) * side_size
            and negative_row == (Disposition.CERTIFIED_ABSENT,) * side_size
        )
        survivor = _bounded_support_admissible(positive_row, negative_row)
        expected_survivors = (
            (self.candidate.candidate_digest,) if survivor else ()
        )
        expected_strict = (
            (self.candidate.candidate_digest,) if strict_survivor else ()
        )
        expected_tier = (
            PanelRubricSupportAcceptanceTier.STRICT_EXACT
            if strict_survivor
            else (
                PanelRubricSupportAcceptanceTier.BOUNDED_ABSTENTION
                if survivor
                else PanelRubricSupportAcceptanceTier.REJECTED
            )
        )
        expected_gap = (
            None
            if survivor
            else _make_gap(
                self.candidate, self.support_panel_ids, self.support_sides, self.row
            )
        )
        if (
            self.survivor_candidate_digests != expected_survivors
            or self.strict_survivor_candidate_digests != expected_strict
            or self.support_acceptance_tier is not expected_tier
            or self.gap != expected_gap
            or self.version_space_digest != canonical_digest(_version_content(self))
        ):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "support survivor, gap, or digest differs"
            )

    def to_data(self) -> dict[str, object]:
        return {**_version_content(self), "version_space_digest": self.version_space_digest}

    @classmethod
    def from_data(
        cls, value: object
    ) -> "ObjectBongardPanelRubricSupportVersionSpace":
        raw = _fields(
            value,
            {
                "schema", "algorithm_id", "algorithm_digest",
                "rubric_spec_digest", "observer_protocol_digest",
                "observer_runtime_identity_digest", "candidate",
                "support_panel_ids", "support_artifact_digests", "support_sides",
                "row", "survivor_candidate_digests", "gap",
                "strict_survivor_candidate_digests", "support_acceptance_tier",
                "support_panels_per_side", "complete_finite_inventory",
                "positive_accept", "negative_accept", *_language_data(),
                "support_policy_digest", *_authority_data(), "version_space_digest",
            },
            "panel rubric support version space",
        )
        if (
            raw["schema"] != PANEL_RUBRIC_VERSION_SPACE_SCHEMA
            or raw["algorithm_id"] != PANEL_RUBRIC_VERSION_SPACE_ALGORITHM_ID
            or raw["support_panels_per_side"] != PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE
            or raw["complete_finite_inventory"] is not True
            or raw["positive_accept"] != Disposition.PRESENT.value
            or raw["negative_accept"] != Disposition.CERTIFIED_ABSENT.value
            or raw["support_policy_digest"]
            != object_bongard_panel_rubric_support_policy_digest()
            or any(raw[key] != item for key, item in _language_data().items())
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(
                not isinstance(raw[name], list)
                for name in (
                    "support_panel_ids", "support_artifact_digests",
                    "support_sides", "row", "survivor_candidate_digests",
                    "strict_survivor_candidate_digests",
                )
            )
        ):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "version-space policy differs"
            )
        try:
            sides = tuple(PanelRubricSupportSide(item) for item in raw["support_sides"])
            row = tuple(Disposition(item) for item in raw["row"])
            tier = PanelRubricSupportAcceptanceTier(
                raw["support_acceptance_tier"]
            )
        except (TypeError, ValueError) as exc:
            raise ObjectBongardPanelRubricVersionSpaceError(
                "support side or disposition is unknown"
            ) from exc
        result = cls(
            raw["algorithm_digest"], raw["rubric_spec_digest"],
            raw["observer_protocol_digest"], raw["observer_runtime_identity_digest"],
            ObjectBongardPanelRubricCandidate.from_data(raw["candidate"]),
            tuple(raw["support_panel_ids"]), tuple(raw["support_artifact_digests"]),
            sides, row, tuple(raw["survivor_candidate_digests"]),
            tuple(raw["strict_survivor_candidate_digests"]), tier,
            (
                None
                if raw["gap"] is None
                else ObjectBongardPanelRubricSupportGap.from_data(raw["gap"])
            ),
            raw["version_space_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricVersionSpaceError(
                "version space is not canonical"
            )
        return result


def _canonical_support(
    values: Sequence[ObjectBongardPanelRubricArtifact], *, label: str
) -> tuple[ObjectBongardPanelRubricArtifact, ...]:
    frozen = tuple(sorted((_validated_artifact(item) for item in values), key=lambda x: x.panel_id))
    if len(frozen) != PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE:
        raise ObjectBongardPanelRubricVersionSpaceError(
            f"{label} support must contain exactly six artifacts"
        )
    return frozen


def _bounded_support_admissible(
    positive_row: Sequence[Disposition], negative_row: Sequence[Disposition]
) -> bool:
    """Apply the frozen 5/6 rule without ordinal sums or error forgiveness."""

    positives = tuple(positive_row)
    negatives = tuple(negative_row)
    side_size = PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE
    if len(positives) != side_size or len(negatives) != side_size:
        raise ObjectBongardPanelRubricVersionSpaceError(
            "bounded support rule requires exactly six rows per side"
        )
    return (
        positives.count(Disposition.PRESENT)
        >= PANEL_RUBRIC_MIN_DEFINITE_MATCHES_PER_SIDE
        and positives.count(Disposition.CERTIFIED_ABSENT) == 0
        and positives.count(Disposition.ERROR) == 0
        and positives.count(Disposition.INDETERMINATE)
        <= PANEL_RUBRIC_MAX_INDETERMINATE_PER_SIDE
        and negatives.count(Disposition.CERTIFIED_ABSENT)
        >= PANEL_RUBRIC_MIN_DEFINITE_MATCHES_PER_SIDE
        and negatives.count(Disposition.PRESENT) == 0
        and negatives.count(Disposition.ERROR) == 0
        and negatives.count(Disposition.INDETERMINATE)
        <= PANEL_RUBRIC_MAX_INDETERMINATE_PER_SIDE
    )


def build_object_bongard_panel_rubric_support_version_space(
    spec: ObjectBongardRubricSpec,
    positives: Sequence[ObjectBongardPanelRubricArtifact],
    negatives: Sequence[ObjectBongardPanelRubricArtifact],
) -> ObjectBongardPanelRubricSupportVersionSpace:
    if not isinstance(spec, ObjectBongardRubricSpec):
        raise TypeError("spec must be ObjectBongardRubricSpec")
    spec = ObjectBongardRubricSpec.from_data(spec.to_data())
    positive_artifacts = _canonical_support(positives, label="positive")
    negative_artifacts = _canonical_support(negatives, label="negative")
    artifacts = positive_artifacts + negative_artifacts
    if len({item.panel_id for item in artifacts}) != len(artifacts):
        raise ObjectBongardPanelRubricVersionSpaceError(
            "support panel IDs must be globally distinct"
        )
    if any(item.rubric_spec != spec for item in artifacts):
        raise ObjectBongardPanelRubricVersionSpaceError(
            "support artifacts do not share the exact rubric spec"
        )
    protocols = {item.protocol_digest for item in artifacts}
    runtimes = {item.runtime_identity_digest for item in artifacts}
    if len(protocols) != 1 or len(runtimes) != 1:
        raise ObjectBongardPanelRubricVersionSpaceError(
            "support artifacts do not share one observer protocol and runtime"
        )
    candidate = enumerate_object_bongard_panel_rubric_candidates(spec)[0]
    sides = (PanelRubricSupportSide.POSITIVE,) * 6 + (
        PanelRubricSupportSide.NEGATIVE,
    ) * 6
    row = tuple(
        evaluate_object_bongard_panel_rubric_candidate(candidate, artifact).disposition
        for artifact in artifacts
    )
    strict_survivor = (
        row[:6] == (Disposition.PRESENT,) * 6
        and row[6:] == (Disposition.CERTIFIED_ABSENT,) * 6
    )
    survivor = _bounded_support_admissible(row[:6], row[6:])
    panel_ids = tuple(item.panel_id for item in artifacts)
    gap = None if survivor else _make_gap(candidate, panel_ids, sides, row)
    values = {
        "algorithm_digest": object_bongard_panel_rubric_version_space_algorithm_digest(),
        "rubric_spec_digest": spec.spec_digest,
        "observer_protocol_digest": next(iter(protocols)),
        "observer_runtime_identity_digest": next(iter(runtimes)),
        "candidate": candidate,
        "support_panel_ids": panel_ids,
        "support_artifact_digests": tuple(item.artifact_digest for item in artifacts),
        "support_sides": sides,
        "row": row,
        "survivor_candidate_digests": (
            (candidate.candidate_digest,) if survivor else ()
        ),
        "strict_survivor_candidate_digests": (
            (candidate.candidate_digest,) if strict_survivor else ()
        ),
        "support_acceptance_tier": (
            PanelRubricSupportAcceptanceTier.STRICT_EXACT
            if strict_survivor
            else (
                PanelRubricSupportAcceptanceTier.BOUNDED_ABSTENTION
                if survivor
                else PanelRubricSupportAcceptanceTier.REJECTED
            )
        ),
        "gap": gap,
    }
    provisional = object.__new__(ObjectBongardPanelRubricSupportVersionSpace)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricSupportVersionSpace(
        **values,  # type: ignore[arg-type]
        version_space_digest=canonical_digest(_version_content(provisional)),
    )


def cold_verify_object_bongard_panel_rubric_support_version_space(
    version_space: ObjectBongardPanelRubricSupportVersionSpace,
    spec: ObjectBongardRubricSpec,
    positives: Sequence[ObjectBongardPanelRubricArtifact],
    negatives: Sequence[ObjectBongardPanelRubricArtifact],
) -> ObjectBongardPanelRubricSupportVersionSpace:
    decoded = ObjectBongardPanelRubricSupportVersionSpace.from_data(
        version_space.to_data()
    )
    replayed = build_object_bongard_panel_rubric_support_version_space(
        spec, positives, negatives
    )
    if decoded != replayed:
        raise ObjectBongardPanelRubricVersionSpaceError(
            "cold version-space replay differs"
        )
    return decoded


__all__ = (
    "ObjectBongardPanelRubricCandidate",
    "ObjectBongardPanelRubricCandidateEvaluation",
    "ObjectBongardPanelRubricSupportDiagnostic",
    "ObjectBongardPanelRubricSupportGap",
    "ObjectBongardPanelRubricSupportVersionSpace",
    "ObjectBongardPanelRubricVersionSpaceError",
    "PANEL_RUBRIC_CANDIDATE_COUNT",
    "PANEL_RUBRIC_MAX_INDETERMINATE_PER_SIDE",
    "PANEL_RUBRIC_MIN_DEFINITE_MATCHES_PER_SIDE",
    "PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE",
    "PANEL_RUBRIC_VERSION_SPACE_ALGORITHM_ID",
    "PanelRubricPredicateOperator",
    "PanelRubricSupportAcceptanceTier",
    "PanelRubricSupportGapKind",
    "PanelRubricSupportSide",
    "build_object_bongard_panel_rubric_support_version_space",
    "cold_verify_object_bongard_panel_rubric_support_version_space",
    "enumerate_object_bongard_panel_rubric_candidates",
    "evaluate_object_bongard_panel_rubric_candidate",
    "object_bongard_panel_rubric_version_space_algorithm_digest",
    "object_bongard_panel_rubric_support_policy_digest",
    "object_bongard_panel_rubric_version_space_source_digest",
)
