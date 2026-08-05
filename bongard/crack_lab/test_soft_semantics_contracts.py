from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

import semantic_legs as L
from semantic_ir import (
    DiagramEdge,
    DiagramSpec,
    LegCall,
    MorphSpec,
    SemanticHypothesis,
)
from semantic_verifier import verify_hypothesis
from dataset import Problem
from soft_semantics import (
    CalibratorContract,
    PrototypeRelationSpec,
    PrototypeRoleSpec,
    PrototypeSpec,
    SoftAbsent,
    SoftError,
    SoftEvidence,
    SoftEvidenceSet,
    content_digest,
    fuzzy_all,
    fuzzy_any,
    fuzzy_max,
    fuzzy_mean,
    fuzzy_min,
    fuzzy_not,
    soft_add,
    soft_pair,
)
from visual_witnesses import AngleWitness, LineSegmentWitness, PointWitness


def _manifest(name: str) -> str:
    return content_digest({"independent_manifest": name})


def _evidence(concept: str, value: float) -> SoftEvidence:
    return SoftEvidence(
        concept_id=concept,
        membership=value,
        producer_digest=_manifest(f"producer-{concept}"),
    )


def _segment(
        source: str, start: tuple[float, float], end: tuple[float, float],
        *, residual: float = 0.0) -> LineSegmentWitness:
    first = PointWitness(x=start[0], y=start[1], source_id=source)
    second = PointWitness(x=end[0], y=end[1], source_id=source)
    return LineSegmentWitness(
        source_component_id=source,
        endpoints=(first, second),
        start=first,
        end=second,
        length=math.dist(start, end),
        residual=residual,
        confidence=max(0.0, 1.0 - residual),
        provenance=(f"fixture-{source}",),
    )


def test_prototype_and_calibrator_are_content_addressed_and_validated():
    roles = (
        PrototypeRoleSpec("body", "PartWitness"),
        PrototypeRoleSpec("appendage", "PartWitness", required=False),
    )
    relation = PrototypeRelationSpec(
        "attached", ("body", "appendage"), "ContactWitness")
    prototype = PrototypeSpec(
        "three-part-icon-v1", roles, (relation,), _manifest("prototype-source"))
    assert prototype.digest() == prototype.digest()
    assert replace(prototype, prototype_id="three-part-icon-v2").digest() \
        != prototype.digest()
    with pytest.raises(ValueError, match="unknown roles"):
        PrototypeSpec(
            "broken-v1", roles,
            (replace(relation, roles=("body", "missing")),),
            _manifest("prototype-source"))

    calibrator = CalibratorContract(
        calibrator_id="external-shape-score-v1",
        metric_id="shape-residual",
        raw_low=0.0,
        raw_high=2.0,
        direction="low",
        score_semantics="similarity",
        source_manifest_digest=_manifest("independent-calibration"),
    )
    best = calibrator.apply(0.0, "shape-similarity")
    worst = calibrator.apply(2.0, "shape-similarity")
    assert isinstance(best, SoftEvidence) and best.membership == 1.0
    assert isinstance(worst, SoftEvidence) and worst.membership == 0.0
    assert best.producer_digest == calibrator.digest()
    assert isinstance(
        calibrator.apply(float("nan"), "shape-similarity"), SoftError)


def test_membership_is_bounded_and_absence_error_are_not_numeric_sentinels():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _evidence("bad-low", -0.01)
    with pytest.raises(ValueError, match="finite"):
        _evidence("bad-nan", float("nan"))

    absent = SoftAbsent("angle-obliqueness", "missing-angle")
    error = SoftError("angle-obliqueness", "fit-failed")
    assert not isinstance(absent, SoftEvidence)
    assert not isinstance(error, SoftEvidence)
    with pytest.raises(L.WitnessAbsent) as failure:
        L.soft_membership_value(absent)
    assert failure.value.failure_mode == "soft_evidence_absent"
    with pytest.raises(ValueError, match="fit-failed"):
        L.soft_membership_value(error)


def test_fuzzy_operators_and_explicit_quantifiers_are_bounded_and_strict():
    low = _evidence("low", 0.2)
    high = _evidence("high", 0.8)
    conjunction = fuzzy_min(low, high)
    assert conjunction.membership == pytest.approx(0.2)
    assert conjunction.input_digests == (low.digest(), high.digest())
    assert fuzzy_max(low, high).membership == pytest.approx(0.8)
    assert fuzzy_not(low).membership == pytest.approx(0.8)

    values = soft_add(soft_pair(low, high), _evidence("middle", 0.5))
    assert values.digest() == values.digest()
    assert fuzzy_all(values).membership == pytest.approx(0.2)
    assert fuzzy_any(values).membership == pytest.approx(0.8)
    assert fuzzy_mean(values).membership == pytest.approx(0.5)
    assert isinstance(fuzzy_all(SoftEvidenceSet(())), SoftAbsent)

    absent = SoftAbsent("missing", "missing-carrier")
    error = SoftError("broken", "extractor-failed")
    assert fuzzy_min(low, absent) is absent
    assert fuzzy_max(low, error) is error


def test_intrinsic_angle_and_analytic_obliqueness_are_honest_and_invariant():
    with pytest.raises(ValueError, match=r"\[0, 180\]"):
        AngleWitness(degrees=181.0)
    horizontal = _segment("a", (0.0, 0.0), (10.0, 0.0))
    diagonal = _segment("b", (0.0, 0.0), (10.0, 10.0))
    vertical = _segment("c", (0.0, 0.0), (0.0, 10.0))

    angle = L.angle_between_segments(horizontal, diagonal)
    assert angle.degrees == pytest.approx(45.0)
    assert angle.reference_frame == "interior"
    evidence = L.angle_obliqueness_evidence(angle)
    assert isinstance(evidence, SoftEvidence)
    assert evidence.membership == pytest.approx(1.0)
    assert L.angle_obliqueness_membership(
        L.angle_between_segments(horizontal, vertical)) == pytest.approx(0.0)

    theta = math.radians(37.0)
    rotation = np.asarray(((math.cos(theta), -math.sin(theta)),
                           (math.sin(theta), math.cos(theta))))
    rotated = []
    for segment in (horizontal, diagonal):
        start = rotation @ np.asarray((segment.start.x, segment.start.y))
        end = rotation @ np.asarray((segment.end.x, segment.end.y))
        rotated.append(_segment(
            segment.source_component_id,
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1]))))
    rotated_angle = L.angle_between_segments(*rotated)
    assert rotated_angle.degrees == pytest.approx(angle.degrees)
    assert L.angle_obliqueness_membership(rotated_angle) \
        == pytest.approx(evidence.membership)

    unrelated = _segment("far", (20.0, 20.0), (30.0, 20.0))
    with pytest.raises(L.WitnessAbsent) as failure:
        L.angle_between_segments(horizontal, unrelated)
    assert failure.value.failure_mode == "segments_do_not_meet"


def test_uncertainty_can_only_weaken_obliqueness_membership():
    exact = L.angle_between_segments(
        _segment("a", (0.0, 0.0), (10.0, 0.0)),
        _segment("b", (0.0, 0.0), (10.0, 10.0)))
    noisy = L.angle_between_segments(
        _segment("a", (0.0, 0.0), (10.0, 0.0), residual=0.03),
        _segment("b", (0.0, 0.0), (10.0, 10.0), residual=0.03))
    assert noisy.uncertainty_degrees > exact.uncertainty_degrees
    assert L.angle_obliqueness_membership(noisy) \
        < L.angle_obliqueness_membership(exact)


def test_registry_exposes_soft_geometry_but_not_open_world_bird_macro():
    assert "angle_obliqueness_membership" not in L.default_registry().names()
    registry = L.soft_semantic_registry()
    names = set(registry.names())
    assert {
        "angle_between_segments",
        "angle_obliqueness_evidence",
        "angle_obliqueness_membership",
        "soft_fuzzy_min",
        "soft_fuzzy_max",
        "soft_fuzzy_not",
        "soft_all",
        "soft_any",
        "soft_mean",
    } <= names
    assert "prototype_bird_like" not in names
    assert all(
        "bird" not in proxy
        for contract in registry.contracts() for proxy in contract.proxy_for)


def test_open_world_bird_prose_still_fails_the_hard_semantic_gate():
    panel = np.zeros((16, 16), dtype=np.uint8)
    panel[4:8, 4:8] = 1
    problem = Problem("soft-gate", "fixture", "", [panel] * 6, [panel] * 6)
    hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id="unsupported-bird",
        description="The figure is bird-like.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("score", LegCall("object_count", ("scene",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=(MorphSpec("translate", "panel"),),
        semantic_requirements=("bird-like",),
    )
    result = verify_hypothesis(hypothesis, L.soft_semantic_registry(), problem)
    assert not result.accepted
    assert result.semantic_issue == "MISSING_LEG"
    assert result.missing_leg is not None
    assert "bird" in result.missing_leg["uncovered_tokens"]
