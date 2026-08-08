"""Focused offline tests for deterministic two-rank panel-rubric selection."""

from __future__ import annotations

import ast
from copy import deepcopy
from functools import lru_cache
import hashlib
from pathlib import Path

import pytest

from bongard.evidence import Disposition
from bongard.object_bongard_panel_rubric_observer import (
    ObjectBongardPanelRubricArtifact,
    observe_object_bongard_panel_rubric,
)
from bongard.object_bongard_panel_rubric_slate import (
    ObjectBongardPanelRubricSlateError,
    ObjectBongardPanelRubricSlateSelection,
    cold_verify_object_bongard_panel_rubric_slate,
    enumerate_object_bongard_panel_rubric_slate,
    select_object_bongard_panel_rubric_slate,
)
from bongard.object_bongard_panel_rubric_version_space import (
    ObjectBongardPanelRubricSupportVersionSpace,
    PanelRubricSupportAcceptanceTier,
    build_object_bongard_panel_rubric_support_version_space,
)
from bongard.object_bongard_rubric_language import ObjectBongardRubricSpec
from bongard.object_bongard_soft_cues import ObjectBongardSoftCue
from bongard.tests.test_prototype_scene_observer import (
    EFFORT,
    LAUNCHER_DIGEST,
    MODEL,
    NO_TOOLS_KWARGS,
    _png,
    _receipt,
)
from bongard.transport import CodexStructuredResult


SEMANTIC_DIGEST = "e" * 64


@lru_cache(maxsize=None)
def _spec(rank: int) -> ObjectBongardRubricSpec:
    cues = (
        (
            "Two curved wedge lobes share one pointed tip.",
            "One rounded lobe touches one triangular lobe.",
        ),
        (
            "A birdlike silhouette carries swept wings.",
            "An angular silhouette carries upright spikes.",
        ),
    )
    target, foil = cues[rank]
    return ObjectBongardRubricSpec.from_soft_cues(
        SEMANTIC_DIGEST,
        ObjectBongardSoftCue.create(target),
        ObjectBongardSoftCue.create(foil),
        rank,
    )


@lru_cache(maxsize=None)
def _artifact(
    rank: int, index: int, disposition: Disposition
) -> ObjectBongardPanelRubricArtifact:
    panel = _png(20 + index)
    panel_id = (
        "bd/bd_panel_rubric_slate_fixture_0000/"
        f"{index // 6}/{index % 6}.png"
    )
    spec = _spec(rank)
    payloads = {
        Disposition.PRESENT: {"lower": 3, "upper": 4},
        Disposition.CERTIFIED_ABSENT: {"lower": 0, "upper": 1},
        Disposition.INDETERMINATE: {"lower": 2, "upper": 2},
    }

    def transport(prompt, paths, names, schema, **_kwargs):
        if disposition is Disposition.ERROR:
            raise RuntimeError("synthetic observer failure")
        payload = payloads[disposition]
        return CodexStructuredResult(
            payload, _receipt(prompt, paths, names, schema, payload)
        )

    return observe_object_bongard_panel_rubric(
        panel,
        panel_id=panel_id,
        rubric_spec=spec,
        expected_panel_sha256=hashlib.sha256(panel).hexdigest(),
        expected_rubric_spec_digest=spec.spec_digest,
        model=MODEL,
        reasoning_effort=EFFORT,
        expected_launcher_digest=LAUNCHER_DIGEST,
        **NO_TOOLS_KWARGS,
        transport=transport,
    )


def _space(
    rank: int,
    positive_states: tuple[Disposition, ...],
    negative_states: tuple[Disposition, ...],
) -> ObjectBongardPanelRubricSupportVersionSpace:
    assert len(positive_states) == len(negative_states) == 6
    positives = tuple(
        _artifact(rank, index, state)
        for index, state in enumerate(positive_states)
    )
    negatives = tuple(
        _artifact(rank, index + 6, state)
        for index, state in enumerate(negative_states)
    )
    return build_object_bongard_panel_rubric_support_version_space(
        _spec(rank), positives, negatives
    )


def _strict(rank: int) -> ObjectBongardPanelRubricSupportVersionSpace:
    return _space(
        rank,
        (Disposition.PRESENT,) * 6,
        (Disposition.CERTIFIED_ABSENT,) * 6,
    )


def _bounded(rank: int) -> ObjectBongardPanelRubricSupportVersionSpace:
    return _space(
        rank,
        (Disposition.PRESENT,) * 5 + (Disposition.INDETERMINATE,),
        (Disposition.CERTIFIED_ABSENT,) * 5 + (Disposition.INDETERMINATE,),
    )


def _rejected(rank: int) -> ObjectBongardPanelRubricSupportVersionSpace:
    return _space(
        rank,
        (Disposition.PRESENT,) * 5 + (Disposition.CERTIFIED_ABSENT,),
        (Disposition.CERTIFIED_ABSENT,) * 6,
    )


def test_rank_zero_bounded_support_beats_rank_one_strict_support() -> None:
    specs = (_spec(0), _spec(1))
    spaces = (_bounded(0), _strict(1))
    selection = select_object_bongard_panel_rubric_slate(specs, spaces)
    candidates = enumerate_object_bongard_panel_rubric_slate(specs)

    assert selection.selected_candidate == candidates[0]
    assert selection.selected_rubric_spec == specs[0]
    assert selection.selected_support_acceptance_tier is (
        PanelRubricSupportAcceptanceTier.BOUNDED_ABSTENTION
    )
    assert selection.selected_has_strict_exact_support is False
    assert selection.bounded_survivor_candidate_digests == (
        candidates[0].candidate_digest,
        candidates[1].candidate_digest,
    )
    assert selection.strict_survivor_candidate_digests == (
        candidates[1].candidate_digest,
    )


def test_rank_one_is_selected_only_after_rank_zero_rejection() -> None:
    specs = (_spec(0), _spec(1))
    spaces = (_rejected(0), _strict(1))
    selection = select_object_bongard_panel_rubric_slate(specs, spaces)

    assert selection.selected_candidate == selection.ordered_candidates[1]
    assert selection.selected_rubric_spec == specs[1]
    assert selection.selected_support_acceptance_tier is (
        PanelRubricSupportAcceptanceTier.STRICT_EXACT
    )
    assert selection.selected_has_strict_exact_support is True
    assert selection.bounded_survivor_candidate_digests == (
        selection.ordered_candidates[1].candidate_digest,
    )


def test_no_bounded_survivor_is_explicit() -> None:
    specs = (_spec(0), _spec(1))
    spaces = (_rejected(0), _rejected(1))
    selection = select_object_bongard_panel_rubric_slate(specs, spaces)

    assert selection.selected_candidate is None
    assert selection.selected_rubric_spec is None
    assert selection.selected_candidate_digest is None
    assert selection.selected_support_acceptance_tier is None
    assert selection.selected_has_strict_exact_support is False
    assert selection.bounded_survivor_candidate_digests == ()
    assert selection.strict_survivor_candidate_digests == ()
    assert selection.to_data()["status"] == "no_bounded_survivor"


def test_cold_replay_round_trip_and_tamper_rejection() -> None:
    specs = (_spec(0), _spec(1))
    spaces = (_bounded(0), _strict(1))
    selection = select_object_bongard_panel_rubric_slate(specs, spaces)

    assert ObjectBongardPanelRubricSlateSelection.from_data(
        selection.to_data()
    ) == selection
    assert cold_verify_object_bongard_panel_rubric_slate(
        selection, specs, spaces
    ) == selection

    tampered = deepcopy(selection.to_data())
    tampered["selected_candidate_digest"] = (
        selection.ordered_candidates[1].candidate_digest
    )
    with pytest.raises(ObjectBongardPanelRubricSlateError):
        ObjectBongardPanelRubricSlateSelection.from_data(tampered)


def test_slate_has_no_atlas_ranker_or_lean_import() -> None:
    source = Path(__file__).parents[1] / "object_bongard_panel_rubric_slate.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    lowered = tuple(item.lower() for item in imports)
    assert not any(
        "atlas" in item or "ranker" in item or "lean" in item
        for item in lowered
    )

