"""Observer-neutral language for prose-grounded Bongard predicates.

The semantic proposer supplies two ordered positive soft cues.  This module
binds their exact bytes into a description-A-versus-description-B rubric, but deliberately
does not bind that rubric to an object atlas, a panel observer, or an ordinal
scale.  The complete-panel ordinal scale is defined alongside the rubric
language so observers and pure-Python evaluators can share one exact meaning
without importing any geometry pipeline.

Prose is data, never executable predicate code.  Python owns the fixed
interval projection.  Lean is absent from identity, decision, and replay.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_bongard_soft_cues import (
    ObjectBongardSoftCue,
    object_bongard_soft_cue_grammar_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


RUBRIC_SPEC_SCHEMA = "gkm.bongard-soft-cue-rubric-spec.v5"
RUBRIC_LANGUAGE_ID = "bongard.soft-cue-rubric/observer-neutral-forward-pair-v2"
RUBRIC_PRESENT_LOWER_BOUND = 3
RUBRIC_ABSENCE_UPPER_BOUND = 1
RUBRIC_ORDINAL_LEVEL_ANCHORS: tuple[tuple[int, str], ...] = (
    (
        0,
        "The complete panel clearly matches description B more aptly "
        "than description A.",
    ),
    (
        1,
        "The complete panel matches description B slightly more aptly "
        "than description A.",
    ),
    (
        2,
        "The complete panel matches both descriptions equally, matches neither "
        "description, or the comparison is genuinely uncertain.",
    ),
    (
        3,
        "The complete panel matches description A slightly more aptly "
        "than description B.",
    ),
    (
        4,
        "The complete panel clearly matches description A more aptly "
        "than description B.",
    ),
)

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ObjectBongardRubricLanguageError(ValueError):
    """A rubric spec, interval, or scale projection is malformed."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "prose_is_observed_not_executable": True,
        "model_can_choose_operator_threshold_or_polarity": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_defines_identity_or_decision": False,
        "lean_required_for_replay": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardRubricLanguageError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardRubricLanguageError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def object_bongard_rubric_language_source_digest() -> str:
    """Return the exact loaded source digest or fail after source mutation."""

    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_bongard_rubric_ordinal_scale_digest() -> str:
    """Content-address the complete-panel ordinal meanings and projection."""

    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-rubric-ordinal-scale.v2",
            "anchors": [list(item) for item in RUBRIC_ORDINAL_LEVEL_ANCHORS],
            "interval_semantics": "inclusive-narrowest-honest-range",
            "present_when_interval_lower_at_least": (
                RUBRIC_PRESENT_LOWER_BOUND
            ),
            "certified_absent_when_interval_upper_at_most": (
                RUBRIC_ABSENCE_UPPER_BOUND
            ),
            "otherwise": Disposition.INDETERMINATE.value,
            "transport_or_parser_failure": Disposition.ERROR.value,
            "failed_fit_is_absence": False,
        }
    )


@dataclass(frozen=True, order=True, slots=True)
class OrdinalLevelInterval:
    """One inclusive interval on the frozen five-level rubric scale."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.lower, bool)
            or isinstance(self.upper, bool)
            or not isinstance(self.lower, int)
            or not isinstance(self.upper, int)
            or not 0 <= self.lower <= self.upper <= 4
        ):
            raise ObjectBongardRubricLanguageError(
                "ordinal interval must lie in 0..4"
            )

    def to_data(self) -> dict[str, int]:
        return {"lower": self.lower, "upper": self.upper}

    @classmethod
    def from_data(cls, value: object) -> "OrdinalLevelInterval":
        raw = _fields(value, {"lower", "upper"}, "ordinal interval")
        return cls(raw["lower"], raw["upper"])


def classify_object_bongard_rubric_interval(
    interval: OrdinalLevelInterval,
) -> Disposition:
    """Project an interval to exactly one of the three non-error states."""

    if not isinstance(interval, OrdinalLevelInterval):
        raise TypeError("interval must be OrdinalLevelInterval")
    if interval.lower >= RUBRIC_PRESENT_LOWER_BOUND:
        return Disposition.PRESENT
    if interval.upper <= RUBRIC_ABSENCE_UPPER_BOUND:
        return Disposition.CERTIFIED_ABSENT
    return Disposition.INDETERMINATE


def object_bongard_soft_contrast_rubric(
    target_cue: ObjectBongardSoftCue,
    foil_cue: ObjectBongardSoftCue,
) -> str:
    """Derive exact model-visible prose from two content-addressed cues."""

    if not isinstance(target_cue, ObjectBongardSoftCue) or not isinstance(
        foil_cue, ObjectBongardSoftCue
    ):
        raise TypeError("target_cue and foil_cue must be typed soft cues")
    target = ObjectBongardSoftCue.from_data(target_cue.to_data())
    foil = ObjectBongardSoftCue.from_data(foil_cue.to_data())
    if target.cue_digest == foil.cue_digest:
        raise ObjectBongardRubricLanguageError(
            "target and foil soft cues must be distinct"
        )
    result = (
        "Judge how much more strongly the visible form matches description A "
        "than description B. "
        f"Description A, {target.text} Description B, {foil.text}"
    )
    if result != result.strip() or len(result.encode("ascii")) > 768:
        raise ObjectBongardRubricLanguageError(
            "derived contrast rubric violates the bounded prose policy"
        )
    return result


def _spec_content(value: "ObjectBongardRubricSpec") -> dict[str, object]:
    # The spec intentionally excludes every observer source and ordinal-scale
    # digest.  Its identity is the ordered prose contrast; an observer artifact
    # binds the independent scale and implementation used to measure it.
    return {
        "schema": RUBRIC_SPEC_SCHEMA,
        "language_id": RUBRIC_LANGUAGE_ID,
        "semantic_artifact_digest": value.semantic_artifact_digest,
        "candidate_rank": value.candidate_rank,
        "target_cue": value.target_cue.to_data(),
        "foil_cue": value.foil_cue.to_data(),
        "rubric": value.rubric,
        "ordered_cue_roles": ["target", "foil"],
        "rubric_derivation_policy": (
            "exact-content-addressed-positive-soft-cue-a-versus-b"
        ),
        "soft_cue_grammar_digest": object_bongard_soft_cue_grammar_digest(),
        "observation_scope_bound_in_spec": False,
        "ordinal_scale_bound_in_spec": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricSpec:
    """One ordered, observer-neutral positive description-A/B contrast."""

    semantic_artifact_digest: str
    candidate_rank: int
    target_cue: ObjectBongardSoftCue
    foil_cue: ObjectBongardSoftCue
    rubric: str
    spec_digest: str

    def __post_init__(self) -> None:
        _digest(self.semantic_artifact_digest, "semantic artifact digest")
        if type(self.candidate_rank) is not int or self.candidate_rank not in (0, 1):
            raise ObjectBongardRubricLanguageError(
                "rubric candidate rank must be zero or one"
            )
        if not isinstance(self.target_cue, ObjectBongardSoftCue) or not isinstance(
            self.foil_cue, ObjectBongardSoftCue
        ):
            raise TypeError("rubric cues must be typed soft cues")
        target = ObjectBongardSoftCue.from_data(self.target_cue.to_data())
        foil = ObjectBongardSoftCue.from_data(self.foil_cue.to_data())
        if (
            target != self.target_cue
            or foil != self.foil_cue
            or self.rubric != object_bongard_soft_contrast_rubric(target, foil)
        ):
            raise ObjectBongardRubricLanguageError(
                "rubric is not the exact ordered soft-cue contrast derivation"
            )
        _digest(self.spec_digest, "rubric spec digest")
        if self.spec_digest != canonical_digest(_spec_content(self)):
            raise ObjectBongardRubricLanguageError("rubric spec digest differs")

    @classmethod
    def from_soft_cues(
        cls,
        semantic_artifact_digest: str,
        target_cue: ObjectBongardSoftCue,
        foil_cue: ObjectBongardSoftCue,
        candidate_rank: int,
    ) -> "ObjectBongardRubricSpec":
        if not isinstance(target_cue, ObjectBongardSoftCue) or not isinstance(
            foil_cue, ObjectBongardSoftCue
        ):
            raise TypeError("rubric cues must be typed soft cues")
        target = ObjectBongardSoftCue.from_data(target_cue.to_data())
        foil = ObjectBongardSoftCue.from_data(foil_cue.to_data())
        values = {
            "semantic_artifact_digest": semantic_artifact_digest,
            "candidate_rank": candidate_rank,
            "target_cue": target,
            "foil_cue": foil,
            "rubric": object_bongard_soft_contrast_rubric(target, foil),
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(
            **values,
            spec_digest=canonical_digest(_spec_content(provisional)),
        )

    @classmethod
    def from_semantic_artifact(
        cls,
        artifact: object,
        *,
        expected_artifact_digest: str,
        candidate_rank: int = 0,
    ) -> "ObjectBongardRubricSpec":
        """Derive one rank from a cold-canonical semantic nomination."""

        # Local import keeps panel observation and replay independent from the
        # proposer transport and its historical object-named implementation.
        from bongard.object_bongard_semantics import ObjectBongardSemanticArtifact
        from bongard.prototype_scene_observer import PrototypeSceneObserverStatus

        expected = _digest(expected_artifact_digest, "expected semantic digest")
        if not isinstance(artifact, ObjectBongardSemanticArtifact):
            raise TypeError("artifact must be ObjectBongardSemanticArtifact")
        try:
            semantic = ObjectBongardSemanticArtifact.from_data(
                artifact.to_data(), expected_artifact_digest=expected
            )
        except Exception as exc:
            raise ObjectBongardRubricLanguageError(
                "semantic artifact is not canonical"
            ) from exc
        if (
            semantic != artifact
            or semantic.artifact_digest != expected
            or semantic.status is not PrototypeSceneObserverStatus.SUCCESS
            or type(candidate_rank) is not int
            or candidate_rank not in (0, 1)
            or len(semantic.soft_cue_candidates) != 2
            or tuple(item.candidate_rank for item in semantic.soft_cue_candidates)
            != (0, 1)
        ):
            raise ObjectBongardRubricLanguageError("semantic artifact differs")
        pair = semantic.soft_cue_candidates[candidate_rank]
        return cls.from_soft_cues(
            expected,
            pair.group_0_cue,
            pair.group_1_cue,
            candidate_rank,
        )

    def to_data(self) -> dict[str, object]:
        return {**_spec_content(self), "spec_digest": self.spec_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricSpec":
        raw = _fields(
            value,
            {
                "schema",
                "language_id",
                "semantic_artifact_digest",
                "candidate_rank",
                "target_cue",
                "foil_cue",
                "rubric",
                "ordered_cue_roles",
                "rubric_derivation_policy",
                "soft_cue_grammar_digest",
                "observation_scope_bound_in_spec",
                "ordinal_scale_bound_in_spec",
                *_authority_data(),
                "spec_digest",
            },
            "rubric spec",
        )
        if (
            raw["schema"] != RUBRIC_SPEC_SCHEMA
            or raw["language_id"] != RUBRIC_LANGUAGE_ID
            or raw["ordered_cue_roles"] != ["target", "foil"]
            or raw["rubric_derivation_policy"]
            != "exact-content-addressed-positive-soft-cue-a-versus-b"
            or raw["soft_cue_grammar_digest"]
            != object_bongard_soft_cue_grammar_digest()
            or raw["observation_scope_bound_in_spec"] is not False
            or raw["ordinal_scale_bound_in_spec"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardRubricLanguageError("rubric spec policy differs")
        result = cls(
            raw["semantic_artifact_digest"],
            raw["candidate_rank"],
            ObjectBongardSoftCue.from_data(raw["target_cue"]),
            ObjectBongardSoftCue.from_data(raw["foil_cue"]),
            raw["rubric"],
            raw["spec_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricLanguageError("rubric spec is not canonical")
        return result


__all__ = (
    "ObjectBongardRubricLanguageError",
    "ObjectBongardRubricSpec",
    "OrdinalLevelInterval",
    "RUBRIC_ABSENCE_UPPER_BOUND",
    "RUBRIC_LANGUAGE_ID",
    "RUBRIC_ORDINAL_LEVEL_ANCHORS",
    "RUBRIC_PRESENT_LOWER_BOUND",
    "RUBRIC_SPEC_SCHEMA",
    "classify_object_bongard_rubric_interval",
    "object_bongard_rubric_language_source_digest",
    "object_bongard_rubric_ordinal_scale_digest",
    "object_bongard_soft_contrast_rubric",
)
