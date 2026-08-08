"""Deterministic selection across two whole-panel soft-rubric candidates.

Each ranked prose contrast contributes exactly one ``PANEL >= 3`` candidate
and one sealed six-plus-six support row.  Python selects rank zero when its
version space admits the candidate under the frozen bounded-abstention policy;
otherwise it selects rank one when admitted.  Exact six-plus-six agreement is
persisted as a diagnostic and never changes that rank order.

There are no ordinal sums, retries, polarity changes, negation, model ranking,
or Lean dependencies.
"""

from __future__ import annotations

from bongard.runtime_source_snapshot import capture_loaded_source, verify_loaded_source


_LOADED_SOURCE_SHA256 = capture_loaded_source(__name__, __file__)

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from bongard.canonical import canonical_digest
from bongard.evidence import Disposition
from bongard.object_bongard_panel_rubric_version_space import (
    ObjectBongardPanelRubricCandidate,
    ObjectBongardPanelRubricSupportVersionSpace,
    PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE,
    PanelRubricSupportAcceptanceTier,
    PanelRubricSupportSide,
    enumerate_object_bongard_panel_rubric_candidates,
    object_bongard_panel_rubric_support_policy_digest,
    object_bongard_panel_rubric_version_space_algorithm_digest,
)
from bongard.object_bongard_rubric_language import ObjectBongardRubricSpec
from bongard.python_predicate_authority import PYTHON_PREDICATE_AUTHORITY_ID


PANEL_RUBRIC_SLATE_SELECTION_SCHEMA = (
    "gkm.bongard-panel-rubric-slate-selection.v1"
)
PANEL_RUBRIC_SLATE_ALGORITHM_ID = (
    "bongard.panel-rubric-slate/two-rank-first-bounded-survivor-v1"
)
PANEL_RUBRIC_SLATE_SPEC_COUNT = 2
PANEL_RUBRIC_SLATE_CANDIDATE_COUNT = 2

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class ObjectBongardPanelRubricSlateError(ValueError):
    """The two-rank panel slate or its replay is malformed."""


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
        "ordinal_sums_allowed": False,
        "retries_allowed": False,
        "model_selects_candidate": False,
    }


def _fields(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    if (
        not isinstance(value, Mapping)
        or any(not isinstance(key, str) for key in value)
        or set(value) != expected
    ):
        raise ObjectBongardPanelRubricSlateError(f"{label} fields differ")
    return value


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ObjectBongardPanelRubricSlateError(
            f"{label} must be a raw lowercase SHA-256"
        )
    return value


def object_bongard_panel_rubric_slate_source_digest() -> str:
    return verify_loaded_source(
        __name__, expected_source_sha256=_LOADED_SOURCE_SHA256
    )


def object_bongard_panel_rubric_slate_algorithm_digest() -> str:
    return canonical_digest(
        {
            "schema": "gkm.bongard-panel-rubric-slate-algorithm.v1",
            "algorithm_id": PANEL_RUBRIC_SLATE_ALGORITHM_ID,
            "implementation_source_sha256": (
                object_bongard_panel_rubric_slate_source_digest()
            ),
            "version_space_algorithm_digest": (
                object_bongard_panel_rubric_version_space_algorithm_digest()
            ),
            "support_policy_digest": (
                object_bongard_panel_rubric_support_policy_digest()
            ),
            "spec_ranks": [0, 1],
            "candidate_order": ["rank-0/panel", "rank-1/panel"],
            "selection_rule": "first-bounded-admissible-in-rank-order",
            "strict_exact_six_plus_six_is_diagnostic_only": True,
            "strict_status_changes_selection": False,
            "dispositions_preserved": [item.value for item in Disposition],
            "support_panels_per_side": PANEL_RUBRIC_SUPPORT_PANELS_PER_SIDE,
            "query_panels_may_enter_selection": False,
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
        len(specs) != PANEL_RUBRIC_SLATE_SPEC_COUNT
        or any(not isinstance(item, ObjectBongardRubricSpec) for item in specs)
        or tuple(item.candidate_rank for item in specs) != (0, 1)
        or len({item.spec_digest for item in specs}) != 2
        or len({item.semantic_artifact_digest for item in specs}) != 1
    ):
        raise ObjectBongardPanelRubricSlateError(
            "panel slate requires distinct canonical rubric ranks zero and one"
        )
    return specs  # type: ignore[return-value]


def enumerate_object_bongard_panel_rubric_slate(
    specs: Sequence[ObjectBongardRubricSpec],
) -> tuple[ObjectBongardPanelRubricCandidate, ObjectBongardPanelRubricCandidate]:
    """Return exactly rank-zero/PANEL then rank-one/PANEL."""

    first, second = _canonical_specs(specs)
    candidates = (
        enumerate_object_bongard_panel_rubric_candidates(first)[0],
        enumerate_object_bongard_panel_rubric_candidates(second)[0],
    )
    if len({item.candidate_digest for item in candidates}) != 2:
        raise ObjectBongardPanelRubricSlateError(
            "ranked panel candidate identities are not distinct"
        )
    return candidates


def _selection_content(
    value: "ObjectBongardPanelRubricSlateSelection",
) -> dict[str, object]:
    return {
        "schema": PANEL_RUBRIC_SLATE_SELECTION_SCHEMA,
        "algorithm_id": PANEL_RUBRIC_SLATE_ALGORITHM_ID,
        "algorithm_digest": value.algorithm_digest,
        "semantic_artifact_digest": value.semantic_artifact_digest,
        "rubric_specs": [item.to_data() for item in value.rubric_specs],
        "version_spaces": [item.to_data() for item in value.version_spaces],
        "ordered_candidates": [item.to_data() for item in value.ordered_candidates],
        "bounded_survivor_candidate_digests": list(
            value.bounded_survivor_candidate_digests
        ),
        "strict_survivor_candidate_digests": list(
            value.strict_survivor_candidate_digests
        ),
        "selected_candidate_digest": value.selected_candidate_digest,
        "selected_support_acceptance_tier": (
            None
            if value.selected_support_acceptance_tier is None
            else value.selected_support_acceptance_tier.value
        ),
        "selected_has_strict_exact_support": (
            value.selected_has_strict_exact_support
        ),
        "status": (
            "selected"
            if value.selected_candidate_digest is not None
            else "no_bounded_survivor"
        ),
        "candidate_order": ["rank-0/panel", "rank-1/panel"],
        "selection_rule": "first-bounded-admissible-in-rank-order",
        "strict_exact_six_plus_six_is_diagnostic_only": True,
        "strict_status_changes_selection": False,
        "support_panels_shared_across_specs": True,
        "all_disposition_rows_bound": True,
        "query_panels_included": False,
        **_authority_data(),
    }


@dataclass(frozen=True, slots=True)
class ObjectBongardPanelRubricSlateSelection:
    """Content-addressed selection over two singleton panel spaces."""

    algorithm_digest: str
    semantic_artifact_digest: str
    rubric_specs: tuple[ObjectBongardRubricSpec, ObjectBongardRubricSpec]
    version_spaces: tuple[
        ObjectBongardPanelRubricSupportVersionSpace,
        ObjectBongardPanelRubricSupportVersionSpace,
    ]
    ordered_candidates: tuple[
        ObjectBongardPanelRubricCandidate,
        ObjectBongardPanelRubricCandidate,
    ]
    bounded_survivor_candidate_digests: tuple[str, ...]
    strict_survivor_candidate_digests: tuple[str, ...]
    selected_candidate_digest: str | None
    selected_support_acceptance_tier: PanelRubricSupportAcceptanceTier | None
    selected_has_strict_exact_support: bool
    selection_digest: str

    def __post_init__(self) -> None:
        if self.algorithm_digest != object_bongard_panel_rubric_slate_algorithm_digest():
            raise ObjectBongardPanelRubricSlateError(
                "panel slate algorithm binding differs"
            )
        _digest(self.semantic_artifact_digest, "semantic artifact digest")
        specs = _canonical_specs(self.rubric_specs)
        if (
            specs != self.rubric_specs
            or self.semantic_artifact_digest != specs[0].semantic_artifact_digest
        ):
            raise ObjectBongardPanelRubricSlateError("slate rubric specs differ")
        if not isinstance(self.version_spaces, tuple) or len(self.version_spaces) != 2:
            raise ObjectBongardPanelRubricSlateError(
                "panel slate requires two version spaces"
            )
        spaces = tuple(
            ObjectBongardPanelRubricSupportVersionSpace.from_data(item.to_data())
            if isinstance(item, ObjectBongardPanelRubricSupportVersionSpace)
            else item
            for item in self.version_spaces
        )
        candidates = enumerate_object_bongard_panel_rubric_slate(specs)
        if (
            any(
                not isinstance(item, ObjectBongardPanelRubricSupportVersionSpace)
                for item in spaces
            )
            or spaces != self.version_spaces
            or tuple(item.rubric_spec_digest for item in spaces)
            != tuple(item.spec_digest for item in specs)
            or len({item.observer_protocol_digest for item in spaces}) != 1
            or len({item.observer_runtime_identity_digest for item in spaces}) != 1
            or spaces[0].support_panel_ids != spaces[1].support_panel_ids
            or spaces[0].support_sides != spaces[1].support_sides
            or self.ordered_candidates != candidates
        ):
            raise ObjectBongardPanelRubricSlateError(
                "panel spaces do not bind one shared rank-ordered support slate"
            )
        bounded = tuple(
            candidate.candidate_digest
            for candidate, space in zip(candidates, spaces, strict=True)
            if candidate.candidate_digest in space.survivor_candidate_digests
        )
        strict_survivors = tuple(
            candidate.candidate_digest
            for candidate, space in zip(candidates, spaces, strict=True)
            if candidate.candidate_digest
            in space.strict_survivor_candidate_digests
        )
        selected = bounded[0] if bounded else None
        selected_index = (
            None
            if selected is None
            else next(
                index
                for index, item in enumerate(candidates)
                if item.candidate_digest == selected
            )
        )
        expected_tier = (
            None
            if selected_index is None
            else spaces[selected_index].support_acceptance_tier
        )
        expected_strict = selected is not None and selected in strict_survivors
        if (
            self.bounded_survivor_candidate_digests != bounded
            or self.strict_survivor_candidate_digests != strict_survivors
            or self.selected_candidate_digest != selected
            or self.selected_support_acceptance_tier is not expected_tier
            or self.selected_has_strict_exact_support is not expected_strict
        ):
            raise ObjectBongardPanelRubricSlateError(
                "panel survivor order, tier, or deterministic selection differs"
            )
        _digest(self.selection_digest, "selection digest")
        if self.selection_digest != canonical_digest(_selection_content(self)):
            raise ObjectBongardPanelRubricSlateError("selection digest differs")

    @property
    def selected_candidate(self) -> ObjectBongardPanelRubricCandidate | None:
        if self.selected_candidate_digest is None:
            return None
        return next(
            item
            for item in self.ordered_candidates
            if item.candidate_digest == self.selected_candidate_digest
        )

    @property
    def selected_rubric_spec(self) -> ObjectBongardRubricSpec | None:
        selected = self.selected_candidate
        if selected is None:
            return None
        return next(
            item
            for item in self.rubric_specs
            if item.spec_digest == selected.rubric_spec_digest
        )

    def to_data(self) -> dict[str, object]:
        return {**_selection_content(self), "selection_digest": self.selection_digest}

    @classmethod
    def from_data(cls, value: object) -> "ObjectBongardPanelRubricSlateSelection":
        raw = _fields(
            value,
            {
                "schema", "algorithm_id", "algorithm_digest",
                "semantic_artifact_digest", "rubric_specs", "version_spaces",
                "ordered_candidates", "bounded_survivor_candidate_digests",
                "strict_survivor_candidate_digests", "selected_candidate_digest",
                "selected_support_acceptance_tier",
                "selected_has_strict_exact_support", "status", "candidate_order",
                "selection_rule", "strict_exact_six_plus_six_is_diagnostic_only",
                "strict_status_changes_selection", "support_panels_shared_across_specs",
                "all_disposition_rows_bound", "query_panels_included",
                *_authority_data(), "selection_digest",
            },
            "panel rubric slate selection",
        )
        for name in (
            "rubric_specs", "version_spaces", "ordered_candidates",
            "bounded_survivor_candidate_digests",
            "strict_survivor_candidate_digests", "candidate_order",
        ):
            if not isinstance(raw[name], list):
                raise ObjectBongardPanelRubricSlateError(
                    f"{name} must be a JSON list"
                )
        if (
            raw["schema"] != PANEL_RUBRIC_SLATE_SELECTION_SCHEMA
            or raw["algorithm_id"] != PANEL_RUBRIC_SLATE_ALGORITHM_ID
            or raw["status"]
            != (
                "selected"
                if raw["selected_candidate_digest"] is not None
                else "no_bounded_survivor"
            )
            or raw["candidate_order"] != ["rank-0/panel", "rank-1/panel"]
            or raw["selection_rule"]
            != "first-bounded-admissible-in-rank-order"
            or raw["strict_exact_six_plus_six_is_diagnostic_only"] is not True
            or raw["strict_status_changes_selection"] is not False
            or raw["support_panels_shared_across_specs"] is not True
            or raw["all_disposition_rows_bound"] is not True
            or raw["query_panels_included"] is not False
            or any(raw[key] != item for key, item in _authority_data().items())
        ):
            raise ObjectBongardPanelRubricSlateError("slate policy differs")
        try:
            tier = (
                None
                if raw["selected_support_acceptance_tier"] is None
                else PanelRubricSupportAcceptanceTier(
                    raw["selected_support_acceptance_tier"]
                )
            )
        except (TypeError, ValueError) as exc:
            raise ObjectBongardPanelRubricSlateError(
                "selected support tier is unknown"
            ) from exc
        result = cls(
            raw["algorithm_digest"],
            raw["semantic_artifact_digest"],
            tuple(ObjectBongardRubricSpec.from_data(item) for item in raw["rubric_specs"]),  # type: ignore[arg-type]
            tuple(
                ObjectBongardPanelRubricSupportVersionSpace.from_data(item)
                for item in raw["version_spaces"]
            ),  # type: ignore[arg-type]
            tuple(
                ObjectBongardPanelRubricCandidate.from_data(item)
                for item in raw["ordered_candidates"]
            ),  # type: ignore[arg-type]
            tuple(raw["bounded_survivor_candidate_digests"]),
            tuple(raw["strict_survivor_candidate_digests"]),
            raw["selected_candidate_digest"],
            tier,
            raw["selected_has_strict_exact_support"],
            raw["selection_digest"],
        )
        if result.to_data() != dict(raw):
            raise ObjectBongardPanelRubricSlateError("slate is not canonical")
        return result


def select_object_bongard_panel_rubric_slate(
    specs: Sequence[ObjectBongardRubricSpec],
    version_spaces: Sequence[ObjectBongardPanelRubricSupportVersionSpace],
) -> ObjectBongardPanelRubricSlateSelection:
    """Select rank zero when bounded-admissible, else rank one."""

    frozen_specs = _canonical_specs(specs)
    spaces = tuple(
        ObjectBongardPanelRubricSupportVersionSpace.from_data(item.to_data())
        if isinstance(item, ObjectBongardPanelRubricSupportVersionSpace)
        else item
        for item in version_spaces
    )
    if len(spaces) != 2 or any(
        not isinstance(item, ObjectBongardPanelRubricSupportVersionSpace)
        for item in spaces
    ):
        raise ObjectBongardPanelRubricSlateError(
            "panel slate requires two canonical version spaces"
        )
    candidates = enumerate_object_bongard_panel_rubric_slate(frozen_specs)
    bounded = tuple(
        candidate.candidate_digest
        for candidate, space in zip(candidates, spaces, strict=True)
        if candidate.candidate_digest in space.survivor_candidate_digests
    )
    strict_survivors = tuple(
        candidate.candidate_digest
        for candidate, space in zip(candidates, spaces, strict=True)
        if candidate.candidate_digest in space.strict_survivor_candidate_digests
    )
    selected = bounded[0] if bounded else None
    selected_index = (
        None
        if selected is None
        else next(
            index
            for index, item in enumerate(candidates)
            if item.candidate_digest == selected
        )
    )
    values = {
        "algorithm_digest": object_bongard_panel_rubric_slate_algorithm_digest(),
        "semantic_artifact_digest": frozen_specs[0].semantic_artifact_digest,
        "rubric_specs": frozen_specs,
        "version_spaces": spaces,
        "ordered_candidates": candidates,
        "bounded_survivor_candidate_digests": bounded,
        "strict_survivor_candidate_digests": strict_survivors,
        "selected_candidate_digest": selected,
        "selected_support_acceptance_tier": (
            None
            if selected_index is None
            else spaces[selected_index].support_acceptance_tier
        ),
        "selected_has_strict_exact_support": (
            selected is not None and selected in strict_survivors
        ),
    }
    provisional = object.__new__(ObjectBongardPanelRubricSlateSelection)
    for name, item in values.items():
        object.__setattr__(provisional, name, item)
    return ObjectBongardPanelRubricSlateSelection(
        **values,  # type: ignore[arg-type]
        selection_digest=canonical_digest(_selection_content(provisional)),
    )


def cold_verify_object_bongard_panel_rubric_slate(
    selection: ObjectBongardPanelRubricSlateSelection,
    specs: Sequence[ObjectBongardRubricSpec],
    version_spaces: Sequence[ObjectBongardPanelRubricSupportVersionSpace],
) -> ObjectBongardPanelRubricSlateSelection:
    """Replay the exact two-row selection with no model transport."""

    if not isinstance(selection, ObjectBongardPanelRubricSlateSelection):
        raise TypeError("selection must be ObjectBongardPanelRubricSlateSelection")
    decoded = ObjectBongardPanelRubricSlateSelection.from_data(selection.to_data())
    replayed = select_object_bongard_panel_rubric_slate(specs, version_spaces)
    if decoded != replayed:
        raise ObjectBongardPanelRubricSlateError("cold panel slate replay differs")
    return decoded


__all__ = (
    "ObjectBongardPanelRubricSlateError",
    "ObjectBongardPanelRubricSlateSelection",
    "PANEL_RUBRIC_SLATE_ALGORITHM_ID",
    "PANEL_RUBRIC_SLATE_CANDIDATE_COUNT",
    "PANEL_RUBRIC_SLATE_SELECTION_SCHEMA",
    "PANEL_RUBRIC_SLATE_SPEC_COUNT",
    "cold_verify_object_bongard_panel_rubric_slate",
    "enumerate_object_bongard_panel_rubric_slate",
    "object_bongard_panel_rubric_slate_algorithm_digest",
    "object_bongard_panel_rubric_slate_source_digest",
    "select_object_bongard_panel_rubric_slate",
)
