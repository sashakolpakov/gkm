"""Typed, content-addressed natural-language cues for Bongard predicates.

The text is observed, never executed.  Python gives it operational meaning by
placing the exact bytes inside a frozen rubric and evaluating the resulting
ordinal intervals with the closed predicate algebra.  Lean is deliberately
absent from identity, evaluation, and replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from bongard.canonical import canonical_digest
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID
from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

SOFT_CUE_SCHEMA = "gkm.bongard-object-soft-cue.v1"
SOFT_CUE_PAIR_SCHEMA = "gkm.bongard-object-soft-cue-pair.v1"
SOFT_CUE_GRAMMAR_ID = "bongard.object-soft-cue/positive-atomic-visible-text-v1"

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ASCII_VISIBLE = re.compile(r"[ -~]+\Z")
_FORBIDDEN_WORD = re.compile(
    r"\b(?:"
    r"no|not|none|neither|nor|never|without|lacks?|lacking|absent|absence|"
    r"missing|omits?|omitted|excludes?|excluding|except|"
    r"and|or|than|versus|different|distinct|unlike|other|"
    r"group|class|label|positive|negative|target|foil|reference|example|"
    r"candidate|proposal|rule|predicate|formula|threshold|score"
    r")\b",
    re.IGNORECASE,
)
_FORBIDDEN_OPERATOR_OR_DIGIT = re.compile(r"[0-9<>=!&|+*/^%]")


class ObjectBongardSoftCueError(ValueError):
    """A soft cue or ranked cue pair is outside the closed prose grammar."""


def _authority_data() -> dict[str, object]:
    return {
        "predicate_authority_id": PYTHON_PREDICATE_AUTHORITY_ID,
        "python_is_canonical_authority": True,
        "cue_text_is_observed_not_executed": True,
        "model_can_choose_operator_threshold_or_polarity": False,
        "negation_allowed": False,
        "polarity_flip_allowed": False,
        "lean_present": False,
        "lean_required": False,
        "lean_removable": True,
        "lean_affects_identity_or_decision": False,
        "lean_required_for_replay": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardSoftCueError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardSoftCueError(f"{label} must be a raw SHA-256")
    return value


def object_bongard_soft_cue_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_bongard_soft_cue_grammar_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-soft-cue-grammar.v1",
            "grammar_id": SOFT_CUE_GRAMMAR_ID,
            "source_digest": object_bongard_soft_cue_source_digest(),
            "minimum_characters": 8,
            "maximum_characters": 240,
            "encoding": "printable-ascii-single-line",
            "exact_whitespace": "stripped-no-tabs-no-double-spaces",
            "forbidden_word_pattern": _FORBIDDEN_WORD.pattern,
            "forbidden_operator_or_digit_pattern": (
                _FORBIDDEN_OPERATOR_OR_DIGIT.pattern
            ),
            "spelled_out_visible_counts_allowed": True,
            "positive_atomic_visible_text_only": True,
            **_authority_data(),
        }
    )


def validate_object_bongard_soft_cue_text(value: object) -> str:
    """Return exact canonical cue text or reject it without normalization."""

    if (
        not isinstance(value, str)
        or not 8 <= len(value) <= 240
        or value != value.strip()
        or _ASCII_VISIBLE.fullmatch(value) is None
        or "  " in value
        or "\t" in value
        or _FORBIDDEN_WORD.search(value) is not None
        or _FORBIDDEN_OPERATOR_OR_DIGIT.search(value) is not None
    ):
        raise ObjectBongardSoftCueError(
            "soft cue text violates the positive atomic visible-text grammar"
        )
    return value


def _cue_content(value: "ObjectBongardSoftCue") -> dict[str, object]:
    return {
        "schema": SOFT_CUE_SCHEMA,
        "grammar_id": SOFT_CUE_GRAMMAR_ID,
        "grammar_digest": object_bongard_soft_cue_grammar_digest(),
        "text": value.text,
        **_authority_data(),
    }


@dataclass(frozen=True, order=True, slots=True)
class ObjectBongardSoftCue:
    text: str
    cue_digest: str

    def __post_init__(self) -> None:
        validate_object_bongard_soft_cue_text(self.text)
        _digest(self.cue_digest, "soft cue digest")
        if self.cue_digest != canonical_digest(_cue_content(self)):
            raise ObjectBongardSoftCueError("soft cue digest differs")

    @classmethod
    def create(cls, text: str) -> "ObjectBongardSoftCue":
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "text", validate_object_bongard_soft_cue_text(text))
        return cls(text, canonical_digest(_cue_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_cue_content(self), "cue_digest": self.cue_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSoftCue":
        raw = _fields(
            value,
            {
                "schema", "grammar_id", "grammar_digest", "text",
                *_authority_data(), "cue_digest",
            },
            "soft cue",
        )
        if (
            raw["schema"] != SOFT_CUE_SCHEMA
            or raw["grammar_id"] != SOFT_CUE_GRAMMAR_ID
            or raw["grammar_digest"] != object_bongard_soft_cue_grammar_digest()
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardSoftCueError("soft cue policy differs")
        result = cls(raw["text"], raw["cue_digest"])
        if result.to_data() != dict(raw):
            raise ObjectBongardSoftCueError("soft cue is not canonical")
        return result


def _pair_content(value: "ObjectBongardSoftCuePair") -> dict[str, object]:
    return {
        "schema": SOFT_CUE_PAIR_SCHEMA,
        "candidate_rank": value.candidate_rank,
        "group_0_cue": value.group_0_cue.to_data(),
        "group_1_cue": value.group_1_cue.to_data(),
        "ordered_group_roles": ["group_0", "group_1"],
        "reverse_orientation_authorized": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSoftCuePair:
    candidate_rank: int
    group_0_cue: ObjectBongardSoftCue
    group_1_cue: ObjectBongardSoftCue
    pair_digest: str

    def __post_init__(self) -> None:
        if type(self.candidate_rank) is not int or self.candidate_rank not in (0, 1):
            raise ObjectBongardSoftCueError("candidate rank must be zero or one")
        if not isinstance(self.group_0_cue, ObjectBongardSoftCue) or not isinstance(
            self.group_1_cue, ObjectBongardSoftCue
        ):
            raise TypeError("soft cue pair members have the wrong type")
        if self.group_0_cue.cue_digest == self.group_1_cue.cue_digest:
            raise ObjectBongardSoftCueError(
                "the two neutral groups require different positive cues"
            )
        _digest(self.pair_digest, "soft cue pair digest")
        if self.pair_digest != canonical_digest(_pair_content(self)):
            raise ObjectBongardSoftCueError("soft cue pair digest differs")

    @classmethod
    def create(
        cls,
        candidate_rank: int,
        group_0_cue: ObjectBongardSoftCue | str,
        group_1_cue: ObjectBongardSoftCue | str,
    ) -> "ObjectBongardSoftCuePair":
        cue_0 = (
            group_0_cue
            if isinstance(group_0_cue, ObjectBongardSoftCue)
            else ObjectBongardSoftCue.create(group_0_cue)
        )
        cue_1 = (
            group_1_cue
            if isinstance(group_1_cue, ObjectBongardSoftCue)
            else ObjectBongardSoftCue.create(group_1_cue)
        )
        values = {
            "candidate_rank": candidate_rank,
            "group_0_cue": cue_0,
            "group_1_cue": cue_1,
        }
        provisional = object.__new__(cls)
        for name, item in values.items():
            object.__setattr__(provisional, name, item)
        return cls(**values, pair_digest=canonical_digest(_pair_content(provisional)))

    def to_data(self) -> dict[str, object]:
        return {**_pair_content(self), "pair_digest": self.pair_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSoftCuePair":
        raw = _fields(
            value,
            {
                "schema", "candidate_rank", "group_0_cue", "group_1_cue",
                "ordered_group_roles", "reverse_orientation_authorized",
                *_authority_data(), "pair_digest",
            },
            "soft cue pair",
        )
        if (
            raw["schema"] != SOFT_CUE_PAIR_SCHEMA
            or raw["ordered_group_roles"] != ["group_0", "group_1"]
            or raw["reverse_orientation_authorized"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardSoftCueError("soft cue pair policy differs")
        result = cls(
            raw["candidate_rank"],
            ObjectBongardSoftCue.from_data(raw["group_0_cue"]),
            ObjectBongardSoftCue.from_data(raw["group_1_cue"]),
            raw["pair_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardSoftCueError("soft cue pair is not canonical")
        return result


__all__ = [
    "ObjectBongardSoftCue",
    "ObjectBongardSoftCueError",
    "ObjectBongardSoftCuePair",
    "SOFT_CUE_GRAMMAR_ID",
    "SOFT_CUE_PAIR_SCHEMA",
    "SOFT_CUE_SCHEMA",
    "object_bongard_soft_cue_grammar_digest",
    "object_bongard_soft_cue_source_digest",
    "validate_object_bongard_soft_cue_text",
]
