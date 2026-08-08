"""Offline tests for the singleton whole-panel rubric version space."""

from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from pathlib import Path

from bongard.evidence import Disposition
from bongard.object_bongard_panel_rubric_observer import (
    observe_object_bongard_panel_rubric,
)
from bongard.object_bongard_panel_rubric_version_space import (
    ObjectBongardPanelRubricCandidate,
    PANEL_RUBRIC_MIN_DEFINITE_MATCHES_PER_SIDE,
    PanelRubricSupportAcceptanceTier,
    PanelRubricSupportGapKind,
    build_object_bongard_panel_rubric_support_version_space,
    cold_verify_object_bongard_panel_rubric_support_version_space,
    enumerate_object_bongard_panel_rubric_candidates,
    evaluate_object_bongard_panel_rubric_candidate,
    object_bongard_panel_rubric_support_policy_digest,
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


def _spec() -> ObjectBongardRubricSpec:
    return ObjectBongardRubricSpec.from_soft_cues(
        "c" * 64,
        ObjectBongardSoftCue.create("Two curved wedge-shaped lobes share one tip."),
        ObjectBongardSoftCue.create("One rounded lobe touches one triangular lobe."),
        0,
    )


@lru_cache(maxsize=None)
def _artifact(index: int, disposition: Disposition):
    panel = _png(70 + index)
    panel_id = f"bd/bd_panel_version_fixture_0000/{index // 6}/{index % 6}.png"
    spec = _spec()
    payload_by_state = {
        Disposition.PRESENT: {"lower": 3, "upper": 4},
        Disposition.CERTIFIED_ABSENT: {"lower": 0, "upper": 1},
        Disposition.INDETERMINATE: {"lower": 2, "upper": 2},
    }

    def transport(prompt, paths, names, schema, **_kwargs):
        if disposition is Disposition.ERROR:
            raise RuntimeError("synthetic observer failure")
        payload = payload_by_state[disposition]
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


def _support(
    positive_states: tuple[Disposition, ...],
    negative_states: tuple[Disposition, ...],
):
    assert len(positive_states) == len(negative_states) == 6
    positives = tuple(_artifact(index, state) for index, state in enumerate(positive_states))
    negatives = tuple(
        _artifact(index + 6, state) for index, state in enumerate(negative_states)
    )
    return positives, negatives


def test_singleton_candidate_and_predicate_false_nomenclature() -> None:
    candidates = enumerate_object_bongard_panel_rubric_candidates(_spec())
    assert len(candidates) == 1
    candidate = candidates[0]
    assert isinstance(candidate, ObjectBongardPanelRubricCandidate)
    assert candidate.scope == "panel"
    assert candidate.threshold == 3
    assert candidate.formula == "PANEL target_preference_level >= 3"
    data = candidate.to_data()
    assert data["certified_absent_observation_meaning"] == "foil_preferred"
    assert data["certified_absent_predicate_meaning"] == (
        "signed-target-preference-predicate-false"
    )
    assert data["literal_absence_of_visual_cue_claimed"] is False
    assert ObjectBongardPanelRubricCandidate.from_data(data) == candidate
    assert len(object_bongard_panel_rubric_support_policy_digest()) == 64


def test_strict_six_plus_six_is_admissible_and_persisted_as_strict() -> None:
    positives, negatives = _support(
        (Disposition.PRESENT,) * 6,
        (Disposition.CERTIFIED_ABSENT,) * 6,
    )
    version = build_object_bongard_panel_rubric_support_version_space(
        _spec(), positives, negatives
    )
    assert version.survivor_candidate_digests == (
        version.candidate.candidate_digest,
    )
    assert version.strict_survivor_candidate_digests == (
        version.candidate.candidate_digest,
    )
    assert version.support_acceptance_tier is (
        PanelRubricSupportAcceptanceTier.STRICT_EXACT
    )
    assert version.gap is None
    assert cold_verify_object_bongard_panel_rubric_support_version_space(
        version, _spec(), positives, negatives
    ) == version


def test_five_of_six_with_one_indeterminate_per_side_is_admissible() -> None:
    assert PANEL_RUBRIC_MIN_DEFINITE_MATCHES_PER_SIDE == 5
    positives, negatives = _support(
        (Disposition.PRESENT,) * 5 + (Disposition.INDETERMINATE,),
        (Disposition.CERTIFIED_ABSENT,) * 5 + (Disposition.INDETERMINATE,),
    )
    version = build_object_bongard_panel_rubric_support_version_space(
        _spec(), positives, negatives
    )
    assert version.survivor_candidate_digests == (
        version.candidate.candidate_digest,
    )
    assert version.strict_survivor_candidate_digests == ()
    assert version.support_acceptance_tier is (
        PanelRubricSupportAcceptanceTier.BOUNDED_ABSTENTION
    )
    assert version.gap is None


def test_confident_contradiction_rejects_as_language_gap() -> None:
    positives, negatives = _support(
        (Disposition.PRESENT,) * 5 + (Disposition.CERTIFIED_ABSENT,),
        (Disposition.CERTIFIED_ABSENT,) * 6,
    )
    version = build_object_bongard_panel_rubric_support_version_space(
        _spec(), positives, negatives
    )
    assert version.survivor_candidate_digests == ()
    assert version.support_acceptance_tier is PanelRubricSupportAcceptanceTier.REJECTED
    assert version.gap is not None
    assert version.gap.kind is PanelRubricSupportGapKind.LANGUAGE_GAP
    assert len(version.gap.diagnostic.definite_counterexample_panel_ids) == 1


def test_error_or_two_abstentions_reject_as_witness_gap() -> None:
    for positives in (
        (Disposition.PRESENT,) * 5 + (Disposition.ERROR,),
        (Disposition.PRESENT,) * 4 + (Disposition.INDETERMINATE,) * 2,
    ):
        positive_artifacts, negatives = _support(
            positives, (Disposition.CERTIFIED_ABSENT,) * 6
        )
        version = build_object_bongard_panel_rubric_support_version_space(
            _spec(), positive_artifacts, negatives
        )
        assert version.survivor_candidate_digests == ()
        assert version.support_acceptance_tier is (
            PanelRubricSupportAcceptanceTier.REJECTED
        )
        assert version.gap is not None
        assert version.gap.kind is PanelRubricSupportGapKind.WITNESS_GAP


def test_evaluation_copies_only_sealed_python_disposition() -> None:
    artifact = _artifact(0, Disposition.CERTIFIED_ABSENT)
    candidate = enumerate_object_bongard_panel_rubric_candidates(_spec())[0]
    evaluation = evaluate_object_bongard_panel_rubric_candidate(
        candidate, artifact
    )
    assert evaluation.disposition is Disposition.CERTIFIED_ABSENT
    assert evaluation.panel_id == artifact.panel_id
    assert evaluation.observer_artifact_digest == artifact.artifact_digest


def test_version_space_has_no_atlas_ranker_or_lean_import() -> None:
    source = Path(__file__).parents[1] / "object_bongard_panel_rubric_version_space.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.append(node.module)
    lowered = tuple(item.lower() for item in imports)
    assert not any("lean" in item or "atlas" in item or "ranker" in item for item in lowered)
    assert "bongard.object_bongard_rubric_observer" not in lowered

