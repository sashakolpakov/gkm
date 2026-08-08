"""Deterministic rank-zero-first selection for shared-witness predicates.

The slate contains exactly the two structured specs produced before support
labels are consulted.  Each spec has one orientation-preserving Python
candidate.  Rank zero wins whenever its fixed support gate survives; rank one
is considered only after rank zero is rejected.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.object_bongard_shared_witness import ObjectBongardSharedWitnessRubricSpec
from bongard.object_bongard_shared_witness_support import (
    ObjectBongardSharedWitnessCandidate,
    ObjectBongardSharedWitnessSupportVersionSpace,
    SharedWitnessSupportAcceptanceTier,
    object_bongard_shared_witness_support_algorithm_digest,
    object_bongard_shared_witness_support_policy_digest,
)
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


SHARED_WITNESS_SLATE_SELECTION_SCHEMA = "gkm.bongard-shared-witness-slate-selection.v1"
SHARED_WITNESS_SLATE_ALGORITHM_ID = (
    "bongard.shared-witness-slate/rank-zero-first-fixed-support-v1"
)
SHARED_WITNESS_SLATE_SPEC_COUNT = 2

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ObjectBongardSharedWitnessSlateError(ValueError):
    """The structured two-rank slate or its replay failed closed."""


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


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardSharedWitnessSlateError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardSharedWitnessSlateError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def object_bongard_shared_witness_slate_source_digest() -> str:
    return verify_loaded_source(__name__, expected_source_sha256=_LOADED_SOURCE_SHA256)


def object_bongard_shared_witness_slate_algorithm_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-shared-witness-slate-algorithm.v1",
            "algorithm_id": SHARED_WITNESS_SLATE_ALGORITHM_ID,
            "implementation_source_sha256": object_bongard_shared_witness_slate_source_digest(),
            "support_algorithm_digest": object_bongard_shared_witness_support_algorithm_digest(),
            "support_policy_digest": object_bongard_shared_witness_support_policy_digest(),
            "spec_rank_order": [0, 1],
            "candidate_order": ["rank-0/group-0-target", "rank-1/group-0-target"],
            "selection_rule": "first-fixed-support-survivor-in-rank-order",
            "query_material_included": False,
            "full_structured_ir_persisted": True,
            "all_entity_observations_persisted_in_spaces": True,
            **_authority_data(),
        }
    )


def _canonical_specs(
    values: Sequence[ObjectBongardSharedWitnessRubricSpec],
) -> tuple[ObjectBongardSharedWitnessRubricSpec, ObjectBongardSharedWitnessRubricSpec]:
    if isinstance(values, (str, bytes)):
        raise TypeError("specs must be a sequence")
    specs = tuple(
        ObjectBongardSharedWitnessRubricSpec.from_data(item.to_data())
        if isinstance(item, ObjectBongardSharedWitnessRubricSpec)
        else item
        for item in values
    )
    if (
        len(specs) != 2
        or any(not isinstance(item, ObjectBongardSharedWitnessRubricSpec) for item in specs)
        or tuple(item.candidate_rank for item in specs) != (0, 1)
        or len({item.spec_digest for item in specs}) != 2
        or len({item.semantic_artifact_digest for item in specs}) != 1
    ):
        raise ObjectBongardSharedWitnessSlateError(
            "slate requires distinct structured rubric ranks zero and one"
        )
    return specs  # type: ignore[return-value]


def enumerate_object_bongard_shared_witness_slate(
    specs: Sequence[ObjectBongardSharedWitnessRubricSpec],
) -> tuple[ObjectBongardSharedWitnessCandidate, ObjectBongardSharedWitnessCandidate]:
    first, second = _canonical_specs(specs)
    candidates = (
        ObjectBongardSharedWitnessCandidate.create(first),
        ObjectBongardSharedWitnessCandidate.create(second),
    )
    if len({item.candidate_digest for item in candidates}) != 2:
        raise ObjectBongardSharedWitnessSlateError("candidate identities are not distinct")
    return candidates


def _selection_content(value: "ObjectBongardSharedWitnessSlateSelection") -> dict[str, object]:
    return {
        "schema": SHARED_WITNESS_SLATE_SELECTION_SCHEMA,
        "algorithm_id": SHARED_WITNESS_SLATE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "semantic_artifact_digest": value.semantic_artifact_digest,
        "rubric_specs": [item.to_data() for item in value.rubric_specs],
        "version_spaces": [item.to_data() for item in value.version_spaces],
        "ordered_candidates": [item.to_data() for item in value.ordered_candidates],
        "survivor_candidate_digests": list(value.survivor_candidate_digests),
        "strict_survivor_candidate_digests": list(value.strict_survivor_candidate_digests),
        "selected_candidate_digest": value.selected_candidate_digest,
        "selected_support_acceptance_tier": (
            None if value.selected_support_acceptance_tier is None
            else value.selected_support_acceptance_tier.value
        ),
        "selected_has_strict_exact_support": value.selected_has_strict_exact_support,
        "status": "selected" if value.selected_candidate_digest is not None else "no_survivor",
        "candidate_order": ["rank-0/group-0-target", "rank-1/group-0-target"],
        "selection_rule": "first-fixed-support-survivor-in-rank-order",
        "rank_zero_selected_when_admissible": True,
        "strict_exact_is_diagnostic_only": True,
        "query_material_included": False,
        "full_structured_ir_persisted": True,
        "all_entity_observations_persisted_in_spaces": True,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardSharedWitnessSlateSelection:
    algorithm_digest: str
    semantic_artifact_digest: str
    rubric_specs: tuple[
        ObjectBongardSharedWitnessRubricSpec,
        ObjectBongardSharedWitnessRubricSpec,
    ]
    version_spaces: tuple[
        ObjectBongardSharedWitnessSupportVersionSpace,
        ObjectBongardSharedWitnessSupportVersionSpace,
    ]
    ordered_candidates: tuple[
        ObjectBongardSharedWitnessCandidate,
        ObjectBongardSharedWitnessCandidate,
    ]
    survivor_candidate_digests: tuple[str, ...]
    strict_survivor_candidate_digests: tuple[str, ...]
    selected_candidate_digest: str | None
    selected_support_acceptance_tier: SharedWitnessSupportAcceptanceTier | None
    selected_has_strict_exact_support: bool
    selection_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != object_bongard_shared_witness_slate_algorithm_digest():
            raise ObjectBongardSharedWitnessSlateError("slate algorithm differs")
        _digest(self.semantic_artifact_digest, "semantic artifact digest")
        specs = _canonical_specs(self.rubric_specs)
        if specs != self.rubric_specs or specs[0].semantic_artifact_digest != self.semantic_artifact_digest:
            raise ObjectBongardSharedWitnessSlateError("slate specs differ")
        if (
            not isinstance(self.version_spaces, tuple)
            or len(self.version_spaces) != 2
            or any(not isinstance(item, ObjectBongardSharedWitnessSupportVersionSpace) for item in self.version_spaces)
            or tuple(item.rubric_spec_digest for item in self.version_spaces)
            != tuple(item.spec_digest for item in specs)
            or len({item.observer_protocol_digest for item in self.version_spaces}) != 1
            or len({item.observer_runtime_identity_digest for item in self.version_spaces}) != 1
        ):
            raise ObjectBongardSharedWitnessSlateError("slate version spaces differ")
        candidates = enumerate_object_bongard_shared_witness_slate(specs)
        if (
            self.ordered_candidates != candidates
            or tuple(item.candidate for item in self.version_spaces) != candidates
        ):
            raise ObjectBongardSharedWitnessSlateError("slate candidate order differs")
        survivors = tuple(
            candidate.candidate_digest
            for candidate, space in zip(candidates, self.version_spaces, strict=True)
            if space.survivor_candidate_digests
        )
        strict = tuple(
            candidate.candidate_digest
            for candidate, space in zip(candidates, self.version_spaces, strict=True)
            if space.strict_survivor_candidate_digests
        )
        selected = survivors[0] if survivors else None
        selected_index = None if selected is None else tuple(item.candidate_digest for item in candidates).index(selected)
        expected_tier = None if selected_index is None else self.version_spaces[selected_index].support_acceptance_tier
        expected_strict = selected is not None and selected in strict
        if (
            self.survivor_candidate_digests != survivors
            or self.strict_survivor_candidate_digests != strict
            or self.selected_candidate_digest != selected
            or self.selected_support_acceptance_tier is not expected_tier
            or self.selected_has_strict_exact_support is not expected_strict
            or self.selection_digest != canonical_digest(_selection_content(self))
        ):
            raise ObjectBongardSharedWitnessSlateError("slate selection or digest differs")

    @property
    def selected_candidate(self) -> ObjectBongardSharedWitnessCandidate | None:
        return next(
            (item for item in self.ordered_candidates if item.candidate_digest == self.selected_candidate_digest),
            None,
        )

    @property
    def selected_rubric_spec(self) -> ObjectBongardSharedWitnessRubricSpec | None:
        selected = self.selected_candidate
        return None if selected is None else self.rubric_specs[selected.candidate_rank]

    @property
    def selected_version_space(self) -> ObjectBongardSharedWitnessSupportVersionSpace | None:
        selected = self.selected_candidate
        return None if selected is None else self.version_spaces[selected.candidate_rank]

    def to_data(self) -> dict[str, object]:
        return {**_selection_content(self), "selection_digest": self.selection_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardSharedWitnessSlateSelection":
        raw = _fields(
            value,
            {
                "schema", "algorithm_id", "algorithm_digest", "semantic_artifact_digest",
                "rubric_specs", "version_spaces", "ordered_candidates",
                "survivor_candidate_digests", "strict_survivor_candidate_digests",
                "selected_candidate_digest", "selected_support_acceptance_tier",
                "selected_has_strict_exact_support", "status", "candidate_order",
                "selection_rule", "rank_zero_selected_when_admissible",
                "strict_exact_is_diagnostic_only", "query_material_included",
                "full_structured_ir_persisted", "all_entity_observations_persisted_in_spaces",
                *_authority_data(), "selection_digest",
            },
            "shared-witness slate selection",
        )
        if (
            raw["schema"] != SHARED_WITNESS_SLATE_SELECTION_SCHEMA
            or raw["algorithm_id"] != SHARED_WITNESS_SLATE_ALGORITHM_ID
            or raw["candidate_order"] != ["rank-0/group-0-target", "rank-1/group-0-target"]
            or raw["selection_rule"] != "first-fixed-support-survivor-in-rank-order"
            or raw["rank_zero_selected_when_admissible"] is not True
            or raw["strict_exact_is_diagnostic_only"] is not True
            or raw["query_material_included"] is not False
            or raw["full_structured_ir_persisted"] is not True
            or raw["all_entity_observations_persisted_in_spaces"] is not True
            or any(raw[key] != item for key, item in _authority_data().items())
            or any(not isinstance(raw[name], list) for name in (
                "rubric_specs", "version_spaces", "ordered_candidates",
                "survivor_candidate_digests", "strict_survivor_candidate_digests",
            ))
        ):
            raise ObjectBongardSharedWitnessSlateError("slate policy differs")
        tier_raw = raw["selected_support_acceptance_tier"]
        try:
            tier = None if tier_raw is None else SharedWitnessSupportAcceptanceTier(tier_raw)
            result = cls(
                raw["algorithm_digest"], raw["semantic_artifact_digest"],
                tuple(ObjectBongardSharedWitnessRubricSpec.from_data(item) for item in raw["rubric_specs"]),
                tuple(ObjectBongardSharedWitnessSupportVersionSpace.from_data(item) for item in raw["version_spaces"]),
                tuple(ObjectBongardSharedWitnessCandidate.from_data(item) for item in raw["ordered_candidates"]),
                tuple(raw["survivor_candidate_digests"]),
                tuple(raw["strict_survivor_candidate_digests"]),
                raw["selected_candidate_digest"], tier,
                raw["selected_has_strict_exact_support"], raw["selection_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise ObjectBongardSharedWitnessSlateError("slate is malformed") from exc
        expected_status = "selected" if result.selected_candidate_digest is not None else "no_survivor"
        if raw["status"] != expected_status or result.to_data() != dict(raw):
            raise ObjectBongardSharedWitnessSlateError("slate is not canonical")
        return result


def select_object_bongard_shared_witness_slate(
    specs: Sequence[ObjectBongardSharedWitnessRubricSpec],
    version_spaces: Sequence[ObjectBongardSharedWitnessSupportVersionSpace],
) -> ObjectBongardSharedWitnessSlateSelection:
    canonical_specs = _canonical_specs(specs)
    if isinstance(version_spaces, (str, bytes)):
        raise TypeError("version_spaces must be a sequence")
    spaces = tuple(
        ObjectBongardSharedWitnessSupportVersionSpace.from_data(item.to_data())
        if isinstance(item, ObjectBongardSharedWitnessSupportVersionSpace)
        else item
        for item in version_spaces
    )
    if len(spaces) != 2 or any(not isinstance(item, ObjectBongardSharedWitnessSupportVersionSpace) for item in spaces):
        raise ObjectBongardSharedWitnessSlateError("slate requires two version spaces")
    candidates = enumerate_object_bongard_shared_witness_slate(canonical_specs)
    survivors = tuple(
        candidate.candidate_digest
        for candidate, space in zip(candidates, spaces, strict=True)
        if space.survivor_candidate_digests
    )
    strict = tuple(
        candidate.candidate_digest
        for candidate, space in zip(candidates, spaces, strict=True)
        if space.strict_survivor_candidate_digests
    )
    selected = survivors[0] if survivors else None
    selected_index = None if selected is None else tuple(item.candidate_digest for item in candidates).index(selected)
    values = {
        "algorithm_digest": object_bongard_shared_witness_slate_algorithm_digest(),
        "semantic_artifact_digest": canonical_specs[0].semantic_artifact_digest,
        "rubric_specs": canonical_specs,
        "version_spaces": spaces,
        "ordered_candidates": candidates,
        "survivor_candidate_digests": survivors,
        "strict_survivor_candidate_digests": strict,
        "selected_candidate_digest": selected,
        "selected_support_acceptance_tier": None if selected_index is None else spaces[selected_index].support_acceptance_tier,
        "selected_has_strict_exact_support": selected is not None and selected in strict,
    }
    provisional = object.__new__(ObjectBongardSharedWitnessSlateSelection)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardSharedWitnessSlateSelection(
        **values, selection_digest=canonical_digest(_selection_content(provisional))
    )


def cold_verify_object_bongard_shared_witness_slate(
    selection: ObjectBongardSharedWitnessSlateSelection,
    specs: Sequence[ObjectBongardSharedWitnessRubricSpec],
    version_spaces: Sequence[ObjectBongardSharedWitnessSupportVersionSpace],
) -> ObjectBongardSharedWitnessSlateSelection:
    decoded = ObjectBongardSharedWitnessSlateSelection.from_data(selection.to_data())
    replayed = select_object_bongard_shared_witness_slate(specs, version_spaces)
    if decoded != replayed:
        raise ObjectBongardSharedWitnessSlateError("cold slate replay differs")
    return decoded


__all__ = (
    "ObjectBongardSharedWitnessSlateError",
    "ObjectBongardSharedWitnessSlateSelection",
    "SHARED_WITNESS_SLATE_ALGORITHM_ID",
    "SHARED_WITNESS_SLATE_SELECTION_SCHEMA",
    "cold_verify_object_bongard_shared_witness_slate",
    "enumerate_object_bongard_shared_witness_slate",
    "object_bongard_shared_witness_slate_algorithm_digest",
    "object_bongard_shared_witness_slate_source_digest",
    "select_object_bongard_shared_witness_slate",
)
