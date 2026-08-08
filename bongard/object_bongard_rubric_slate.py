"""Pure-Python selection across two ranked soft-cue rubric specifications.

Each rubric specification contributes the complete closed OBJECT/SCENE
version space evaluated on the same six positive and six negative support
panels.  This module concatenates those inventories in one fixed rank-major
order and selects the first exact support survivor.  It cannot reinterpret an
indeterminate or failed observation as absence, tune a threshold, reverse a
polarity, or inspect query pixels.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_bongard_rubric_observer import (
    ObjectBongardRubricSpec,
    RubricScope,
)
from bongard.object_bongard_rubric_version_space import (
    ObjectBongardRubricCandidate,
    ObjectBongardRubricSupportVersionSpace,
    RubricSupportSide,
    enumerate_object_bongard_rubric_candidates,
    object_bongard_rubric_version_space_algorithm_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


RUBRIC_SLATE_SELECTION_SCHEMA = "gkm.bongard-object-rubric-slate-selection.v1"
RUBRIC_SLATE_ALGORITHM_ID = (
    "bongard.object-rubric-slate/two-ranked-six-plus-six-first-survivor-v1"
)
RUBRIC_SLATE_SPEC_COUNT = 2
RUBRIC_SLATE_CANDIDATE_COUNT = 4
RUBRIC_SLATE_SUPPORT_PER_SIDE = 6

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ObjectBongardRubricSlateError(ValueError):
    """The two-spec support slate or its canonical replay is malformed."""


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
        "model_selects_scope_or_candidate": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardRubricSlateError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardRubricSlateError(
            f"{label} must be a lowercase raw SHA-256"
        )
    return value


def object_bongard_rubric_slate_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_bongard_rubric_slate_algorithm_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-object-rubric-slate-algorithm.v1",
            "algorithm_id": RUBRIC_SLATE_ALGORITHM_ID,
            "implementation_source_sha256": (
                object_bongard_rubric_slate_source_digest()
            ),
            "rubric_version_space_algorithm_digest": (
                object_bongard_rubric_version_space_algorithm_digest()
            ),
            "spec_ranks": [0, 1],
            "candidate_order": [
                "rank-0/object",
                "rank-0/scene",
                "rank-1/object",
                "rank-1/scene",
            ],
            "support_per_side": RUBRIC_SLATE_SUPPORT_PER_SIDE,
            "survivor_rule": (
                "all-six-positive-present-and-all-six-negative-certified-absent"
            ),
            "selection_rule": "first-survivor-in-frozen-rank-major-order",
            "failed_indeterminate_or_error_is_absence": False,
            "dispositions_preserved": [
                Disposition.PRESENT.value,
                Disposition.CERTIFIED_ABSENT.value,
                Disposition.INDETERMINATE.value,
                Disposition.ERROR.value,
            ],
            "query_or_broad_panels_may_enter_selection": False,
            **_authority_data(),
        }
    )


def _canonical_specs(
    values: Sequence[ObjectBongardRubricSpec],
) -> tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec]:
    specs = tuple(
        ObjectBongardRubricSpec.from_data(item.to_data())
        if isinstance(item, ObjectBongardRubricSpec)
        else item
        for item in values
    )
    if (
        len(specs) != RUBRIC_SLATE_SPEC_COUNT
        or any(not isinstance(item, ObjectBongardRubricSpec) for item in specs)
        or tuple(item.candidate_rank for item in specs) != (0, 1)
        or len({item.spec_digest for item in specs}) != RUBRIC_SLATE_SPEC_COUNT
        or len({item.semantic_artifact_digest for item in specs}) != 1
    ):
        raise ObjectBongardRubricSlateError(
            "slate requires two distinct canonical rubric specs at ranks zero and one"
        )
    return specs  # type: ignore[return-value]


def enumerate_object_bongard_rubric_slate(
    specs: Sequence[ObjectBongardRubricSpec],
) -> tuple[ObjectBongardRubricCandidate, ...]:
    """Return rank-0 OBJECT/SCENE then rank-1 OBJECT/SCENE."""

    first, second = _canonical_specs(specs)
    candidates = (
        *enumerate_object_bongard_rubric_candidates(first),
        *enumerate_object_bongard_rubric_candidates(second),
    )
    if (
        len(candidates) != RUBRIC_SLATE_CANDIDATE_COUNT
        or tuple(item.scope for item in candidates)
        != (
            RubricScope.OBJECT,
            RubricScope.SCENE,
            RubricScope.OBJECT,
            RubricScope.SCENE,
        )
    ):
        raise ObjectBongardRubricSlateError(
            "upstream candidate inventory differs from the frozen slate"
        )
    return candidates


def _selection_content(
    value: "ObjectBongardRubricSlateSelection",
) -> dict[str, object]:
    return {
        "schema": RUBRIC_SLATE_SELECTION_SCHEMA,
        "algorithm_id": RUBRIC_SLATE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "semantic_artifact_digest": value.semantic_artifact_digest,
        "rubric_specs": [item.to_data() for item in value.rubric_specs],
        "version_spaces": [item.to_data() for item in value.version_spaces],
        "ordered_candidates": [item.to_data() for item in value.ordered_candidates],
        "survivor_candidate_digests": list(value.survivor_candidate_digests),
        "selected_candidate_digest": value.selected_candidate_digest,
        "status": "selected" if value.selected_candidate_digest is not None else "no_exact_survivor",
        "candidate_order": [
            "rank-0/object",
            "rank-0/scene",
            "rank-1/object",
            "rank-1/scene",
        ],
        "support_per_side": RUBRIC_SLATE_SUPPORT_PER_SIDE,
        "support_panels_shared_across_specs": True,
        "all_disposition_rows_bound_in_version_spaces": True,
        "positive_accept": "present",
        "negative_accept": "certified_absent",
        "failed_indeterminate_or_error_is_absence": False,
        "dispositions_preserved": [
            Disposition.PRESENT.value,
            Disposition.CERTIFIED_ABSENT.value,
            Disposition.INDETERMINATE.value,
            Disposition.ERROR.value,
        ],
        "selection_rule": "first-survivor-in-frozen-rank-major-order",
        "fallback_or_polarity_rescue_allowed": False,
        "query_or_broad_panels_included": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardRubricSlateSelection:
    """Content-addressed selection over two complete support version spaces."""

    algorithm_digest: str
    semantic_artifact_digest: str
    rubric_specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec]
    version_spaces: tuple[
        ObjectBongardRubricSupportVersionSpace,
        ObjectBongardRubricSupportVersionSpace,
    ]
    ordered_candidates: tuple[ObjectBongardRubricCandidate, ...]
    survivor_candidate_digests: tuple[str, ...]
    selected_candidate_digest: str | None
    selection_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != object_bongard_rubric_slate_algorithm_digest():
            raise ObjectBongardRubricSlateError("slate algorithm binding differs")
        _digest(self.semantic_artifact_digest, "semantic artifact digest")
        specs = _canonical_specs(self.rubric_specs)
        if specs != self.rubric_specs or self.semantic_artifact_digest != specs[0].semantic_artifact_digest:
            raise ObjectBongardRubricSlateError("slate rubric specs differ")
        if not isinstance(self.version_spaces, tuple) or len(self.version_spaces) != 2:
            raise ObjectBongardRubricSlateError("slate requires two version spaces")
        spaces = tuple(
            ObjectBongardRubricSupportVersionSpace.from_data(item.to_data())
            if isinstance(item, ObjectBongardRubricSupportVersionSpace)
            else item
            for item in self.version_spaces
        )
        if any(not isinstance(item, ObjectBongardRubricSupportVersionSpace) for item in spaces):
            raise TypeError("slate version spaces have the wrong type")
        expected_candidates = enumerate_object_bongard_rubric_slate(specs)
        if (
            spaces != self.version_spaces
            or tuple(item.rubric_spec_digest for item in spaces)
            != tuple(item.spec_digest for item in specs)
            or len({item.observer_catalog_digest for item in spaces}) != 1
            or len({item.observer_runtime_identity_digest for item in spaces}) != 1
            or spaces[0].support_panel_ids != spaces[1].support_panel_ids
            or spaces[0].support_sides != spaces[1].support_sides
            or spaces[0].support_sides
            != (RubricSupportSide.POSITIVE,) * RUBRIC_SLATE_SUPPORT_PER_SIDE
            + (RubricSupportSide.NEGATIVE,) * RUBRIC_SLATE_SUPPORT_PER_SIDE
            or self.ordered_candidates != expected_candidates
        ):
            raise ObjectBongardRubricSlateError(
                "version spaces do not bind one shared canonical six-plus-six support slate"
            )
        expected_survivors = tuple(
            candidate.candidate_digest
            for candidate, space in zip(
                expected_candidates,
                (spaces[0], spaces[0], spaces[1], spaces[1]),
                strict=True,
            )
            if candidate.candidate_digest in space.survivor_candidate_digests
        )
        expected_selected = expected_survivors[0] if expected_survivors else None
        if (
            self.survivor_candidate_digests != expected_survivors
            or self.selected_candidate_digest != expected_selected
        ):
            raise ObjectBongardRubricSlateError(
                "slate survivor order or deterministic selection differs"
            )
        _digest(self.selection_digest, "selection digest")
        if self.selection_digest != canonical_digest(_selection_content(self)):
            raise ObjectBongardRubricSlateError("selection digest differs")

    @property
    def selected_candidate(self) -> ObjectBongardRubricCandidate | None:
        if self.selected_candidate_digest is None:
            return None
        return next(
            item
            for item in self.ordered_candidates
            if item.candidate_digest == self.selected_candidate_digest
        )

    def to_data(self) -> dict[str, object]:
        return {**_selection_content(self), "selection_digest": self.selection_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardRubricSlateSelection":
        raw = _fields(
            value,
            {
                "schema", "algorithm_id", "algorithm_digest",
                "semantic_artifact_digest", "rubric_specs", "version_spaces",
                "ordered_candidates", "survivor_candidate_digests",
                "selected_candidate_digest", "status", "candidate_order",
                "support_per_side", "support_panels_shared_across_specs",
                "all_disposition_rows_bound_in_version_spaces", "positive_accept",
                "negative_accept", "failed_indeterminate_or_error_is_absence",
                "dispositions_preserved",
                "selection_rule", "fallback_or_polarity_rescue_allowed",
                "query_or_broad_panels_included", *_authority_data(),
                "selection_digest",
            },
            "rubric slate selection",
        )
        for name in (
            "rubric_specs", "version_spaces", "ordered_candidates",
            "survivor_candidate_digests", "candidate_order",
            "dispositions_preserved",
        ):
            if not isinstance(raw[name], list):
                raise ObjectBongardRubricSlateError(f"{name} must be a JSON list")
        result = cls(
            raw["algorithm_digest"],
            raw["semantic_artifact_digest"],
            tuple(ObjectBongardRubricSpec.from_data(item) for item in raw["rubric_specs"]),
            tuple(ObjectBongardRubricSupportVersionSpace.from_data(item) for item in raw["version_spaces"]),
            tuple(ObjectBongardRubricCandidate.from_data(item) for item in raw["ordered_candidates"]),
            tuple(raw["survivor_candidate_digests"]),
            raw["selected_candidate_digest"],
            raw["selection_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardRubricSlateError("rubric slate is not canonical")
        return result


def select_object_bongard_rubric_slate(
    specs: Sequence[ObjectBongardRubricSpec],
    version_spaces: Sequence[ObjectBongardRubricSupportVersionSpace],
) -> ObjectBongardRubricSlateSelection:
    """Select the first exact survivor without opening any held-out panel."""

    canonical_specs = _canonical_specs(specs)
    spaces = tuple(
        ObjectBongardRubricSupportVersionSpace.from_data(item.to_data())
        if isinstance(item, ObjectBongardRubricSupportVersionSpace)
        else item
        for item in version_spaces
    )
    candidates = enumerate_object_bongard_rubric_slate(canonical_specs)
    if len(spaces) != 2 or any(
        not isinstance(item, ObjectBongardRubricSupportVersionSpace)
        for item in spaces
    ):
        raise ObjectBongardRubricSlateError("slate requires two canonical version spaces")
    survivors = tuple(
        candidate.candidate_digest
        for candidate, space in zip(
            candidates, (spaces[0], spaces[0], spaces[1], spaces[1]), strict=True
        )
        if candidate.candidate_digest in space.survivor_candidate_digests
    )
    values = {
        "algorithm_digest": object_bongard_rubric_slate_algorithm_digest(),
        "semantic_artifact_digest": canonical_specs[0].semantic_artifact_digest,
        "rubric_specs": canonical_specs,
        "version_spaces": spaces,
        "ordered_candidates": candidates,
        "survivor_candidate_digests": survivors,
        "selected_candidate_digest": survivors[0] if survivors else None,
    }
    provisional = object.__new__(ObjectBongardRubricSlateSelection)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardRubricSlateSelection(
        **values,  # type: ignore[arg-type]
        selection_digest=canonical_digest(_selection_content(provisional)),
    )


def cold_verify_object_bongard_rubric_slate(
    selection: ObjectBongardRubricSlateSelection,
    specs: Sequence[ObjectBongardRubricSpec],
    version_spaces: Sequence[ObjectBongardRubricSupportVersionSpace],
) -> ObjectBongardRubricSlateSelection:
    """Replay the selector from canonical support artifacts without a model call."""

    if not isinstance(selection, ObjectBongardRubricSlateSelection):
        raise TypeError("selection must be ObjectBongardRubricSlateSelection")
    decoded = ObjectBongardRubricSlateSelection.from_data(selection.to_data())
    replayed = select_object_bongard_rubric_slate(specs, version_spaces)
    if decoded != replayed:
        raise ObjectBongardRubricSlateError("cold slate replay differs")
    return decoded


__all__ = (
    "ObjectBongardRubricSlateError",
    "ObjectBongardRubricSlateSelection",
    "RUBRIC_SLATE_ALGORITHM_ID",
    "RUBRIC_SLATE_CANDIDATE_COUNT",
    "RUBRIC_SLATE_SELECTION_SCHEMA",
    "RUBRIC_SLATE_SPEC_COUNT",
    "RUBRIC_SLATE_SUPPORT_PER_SIDE",
    "cold_verify_object_bongard_rubric_slate",
    "enumerate_object_bongard_rubric_slate",
    "object_bongard_rubric_slate_algorithm_digest",
    "object_bongard_rubric_slate_source_digest",
    "select_object_bongard_rubric_slate",
)
