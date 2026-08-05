import copy
import json
import math
import os
import re
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import Problem, _draw_polyline, trace_shape
from semantic_compiler import CompileError, compile_hypothesis
from semantic_ir import DiagramEdge, DiagramSpec, LegCall, MorphSpec, SemanticHypothesis
from semantic_legs import LegContract, LegRegistry, WitnessAbsent, default_registry
from semantic_requirements import (
    calibrated_claim_signature,
    leg_suggestions,
    parse_score_operator,
    term_matches_contract_claim,
)
import semantic_legs as L
import semantic_artifacts as SA
import semantic_replay as SR
import phase_d_protocol as PD
from semantic_verifier import verify_hypothesis
from semantic_selection import CandidateEvaluation, ComplexityBreakdown, RiskVector, Track, pareto_frontier
from cofibrations import CofibrationSpec, verify_cofibration
from cofibered_proposer import (
    AnthropicCofiberedProposer,
    ProposalBundle,
    build_prompt,
    hypotheses_from_tool_input,
)
from replay_semantic_runspec import verifier_related_sources
from run_semantic_cone import (
    ProblemResult,
    _checkpoint_payload,
    _load_resume_state,
    _misses,
    _publish_phase_d_track_report,
    _result_payload,
    _replay_terminal_record,
    _select,
    _selection_evidence,
    _selection_record,
    _status_of,
    _terminal_evidence,
    _write_replay_spec,
)
from visual_witnesses import (
    ContactWitness,
    ContourWitness,
    PartGraphWitness,
    PartWitness,
    PointWitness,
)

SQUARE = (
    "line_normal_0.500-0.500",
    "line_normal_0.500-0.750",
    "line_normal_0.500-0.750",
    "line_normal_0.500-0.750",
)
CIRCLE = ("arc_normal_0.300_1.000-0.500",)


def _problem_two_objects_vs_one() -> Problem:
    def panel(two: bool, offset: int) -> np.ndarray:
        arr = np.zeros((128, 128), dtype=np.uint8)
        arr[24 + offset:42 + offset, 24:42] = 1
        if two:
            arr[74 - offset:92 - offset, 82:100] = 1
        return arr

    pos = tuple(panel(True, i) for i in range(6))
    neg = tuple(panel(False, i) for i in range(6))
    return Problem("fixture", "fixture", "two_objects_vs_one", pos, neg)


def _morphism() -> tuple[MorphSpec, ...]:
    return (MorphSpec("translate", "panel"),)


def _object_count_hypothesis() -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id="two_principal_objects",
        description="Positive panels have a higher object count than negatives.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("score", LegCall("object_count", ("scene",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("higher object count",),
    )


def _raw_ink_hypothesis() -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id="more_ink",
        description="Positive panels have more total ink.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("score", LegCall("total_ink", ("panel",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("more total ink",),
    )


def _triangle_proxy_hypothesis() -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id="triangle_by_aspect_proxy",
        description="The figure contains a triangle attached to a quadrilateral.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge("score", LegCall("bbox_aspect", ("main",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("triangle", "quadrilateral", "attachment"),
    )


def _triangle_witness_hypothesis() -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id="triangle_witness_path",
        description="The principal object is triangular.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge("contour", LegCall("extract_contours", ("main",))),
            DiagramEdge("polygon", LegCall("fit_polygon", ("contour",))),
            DiagramEdge("triangle", LegCall("classify_triangle", ("polygon",))),
            DiagramEdge("score", LegCall("witness_confidence", ("triangle",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("triangle",),
        witness_requirements=("TriangleWitness",),
    )


def _attachment_hypothesis() -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id="typed_part_gluing",
        description="The principal parts are attached.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("graph", LegCall("build_part_graph", ("scene",))),
            DiagramEdge(
                "principal_part",
                LegCall("select_largest_part", ("graph",))),
            DiagramEdge(
                "attachment",
                LegCall("detect_attachment", ("graph",))),
            DiagramEdge(
                "score", LegCall("contact_confidence", ("attachment",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=(MorphSpec("translate", "panel"),),
        semantic_requirements=("parts attached",),
        cofibrations=(CofibrationSpec(
            name="part_glued_into_graph",
            source_node="principal_part",
            target_node="graph",
            source_type="PartWitness",
            target_type="PartGraphWitness",
            interface_fields=("contacts",),
            added_fields=("parts",),
            attachment_leg="detect_attachment",
            projection_leg="select_largest_part",
        ),),
    )


def _terminal_candidate_fixture(
        hypothesis: SemanticHypothesis | None = None) \
        -> tuple[Problem, ProblemResult]:
    """Build one offline terminal record from exact verifier evidence."""
    source = _problem_two_objects_vs_one()
    problem = Problem(
        source.problem_id, "basic", source.concept, source.pos, source.neg)
    hypothesis = hypothesis or _object_count_hypothesis()
    verification = verify_hypothesis(
        hypothesis, default_registry(), problem)
    origins = [{
        "round": 0,
        "round_candidate_index": 0,
        "round_candidate_count": 1,
    }]
    rounds = [{
        "round": 0,
        "proposer_kind": "offline-test",
        "parse_error": "",
        "candidate_count": 1,
        "candidate_ids": [hypothesis.hypothesis_id],
        "hypothesis_digests": [
            SR.semantic_cone_digest(hypothesis.to_dict())],
    }]
    evidence = _terminal_evidence(
        rounds, [verification], [hypothesis.to_dict()], verification,
        0.02, origins)
    status = _status_of(verification, ProposalBundle(
        problem_id="problem_00",
        hypotheses=(hypothesis,),
        raw_text="offline fixture",
        proposer_kind="offline-test",
    ))
    solved = status.startswith("SOLVED_SEMANTIC_PURE")
    selection = evidence["selection"]
    record = ProblemResult(
        opaque_id="problem_00",
        category="basic",
        solved=solved,
        selected_hypothesis=hypothesis.hypothesis_id,
        selected_description=hypothesis.description,
        selected_rule=verification.rule,
        support_errors=verification.support_errors,
        loo_errors=verification.loo_errors,
        rotated_loo_errors=verification.rotated_loo_errors,
        rotated_loo_checks=verification.rotated_loo_checks,
        n_examples=verification.n_examples,
        complexity=verification.complexity,
        rounds_used=1,
        proposer_kind="offline-test",
        track="SEMANTIC-PURE",
        condition=PD.OBSERVED,
        sharing_policy=PD.SHARED,
        corpus_digest="sha256:" + "1" * 64,
        panel_set_digest=SR.panel_set_digest(
            SR.panel_records_from_problem(problem)),
        control_digest="",
        status=status,
        proposer_error="",
        candidates=[verification.to_dict()],
        candidate_manifest=selection["candidate_manifest"],
        selection=selection["selected_record"],
        terminal_evidence=evidence,
        terminal_evidence_digest=SR.canonical_json_digest(evidence),
        replay_spec_digest=("sha256:" + "2" * 64 if solved else ""),
    )
    return problem, record


def test_trace_square_closes():
    x, y = trace_shape(SQUARE)[-1]
    assert abs(x) < 1e-9
    assert abs(y) < 1e-9


def test_human_like_object_count_cone_solves_fixture():
    result = verify_hypothesis(
        _object_count_hypothesis(),
        default_registry(),
        _problem_two_objects_vs_one(),
    )
    assert result.accepted
    assert result.support_errors == 0
    assert result.loo_errors == 0
    assert result.semantic_issue == ""
    assert result.rule.startswith("score>=")


def test_leave_one_out_counts_each_panel_once():
    result = verify_hypothesis(
        _object_count_hypothesis(),
        default_registry(),
        _problem_two_objects_vs_one(),
    )
    assert result.n_examples == 12
    assert result.support_accuracy == 1.0
    assert result.loo_accuracy == 1.0
    assert result.rotated_loo_accuracy == 1.0
    assert result.rotated_loo_errors == 0
    assert result.rotated_loo_checks == 72


def test_direct_panel_measurement_is_not_semantic_pure():
    result = verify_hypothesis(
        _raw_ink_hypothesis(),
        default_registry(),
        _problem_two_objects_vs_one(),
    )
    assert not result.accepted
    assert result.semantic_issue == "measurement_only_direct_panel_statistic"


def test_rich_semantic_terms_cannot_compile_to_bbox_proxy():
    result = verify_hypothesis(
        _triangle_proxy_hypothesis(),
        default_registry(),
        _problem_two_objects_vs_one(),
    )
    assert not result.accepted
    assert result.rule == "MISSING_LEG"
    assert result.semantic_issue == "MISSING_LEG"
    assert result.missing_leg["semantic_term"] == "triangle"
    assert "TriangleWitness" in result.missing_leg["required_witness_types"]


def test_relative_measurements_cannot_launder_categorical_adjectives():
    object_prefix = (
        DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
        DiagramEdge("main", LegCall("select_largest", ("scene",))),
    )
    skeleton_prefix = object_prefix + (
        DiagramEdge(
            "skeleton", LegCall("build_skeleton_graph", ("main",))),
    )
    cases = (
        ("thin", object_prefix + (
            DiagramEdge("score", LegCall("bbox_aspect", ("main",))),)),
        ("filled", object_prefix + (
            DiagramEdge("score", LegCall("bbox_occupancy", ("main",))),)),
        ("connected", (
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("score", LegCall("object_count", ("scene",))),
        )),
        ("closed", skeleton_prefix + (
            DiagramEdge("score", LegCall("endpoint_count", ("skeleton",))),)),
        ("acyclic", skeleton_prefix + (
            DiagramEdge("score", LegCall("cycle_count", ("skeleton",))),)),
    )
    for term, edges in cases:
        hypothesis = SemanticHypothesis(
            version="0.1",
            hypothesis_id=f"categorical_proxy_{term}",
            description=f"The object is {term}.",
            polarity="positive_satisfies",
            diagram=DiagramSpec(edges),
            score_node="score",
            order="low_positive",
            preservation_morphisms=_morphism(),
            semantic_requirements=(term,),
        )
        result = verify_hypothesis(
            hypothesis, default_registry(), _problem_two_objects_vs_one())
        assert result.rule == "MISSING_LEG", term
        assert result.semantic_issue == "MISSING_LEG", term


def test_triangle_semantics_require_primitive_witness_path():
    cone = compile_hypothesis(_triangle_witness_hypothesis(), default_registry())
    assert cone.node_types["contour"] == "ContourWitness"
    assert cone.node_types["polygon"] == "PolygonWitness"
    assert cone.node_types["triangle"] == "TriangleWitness"
    assert "triangle" in cone.node_dependencies["score"]


def test_node_and_leg_name_collision_cannot_launder_decorative_witness():
    hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id="dependency_namespace_collision",
        description="The principal object is triangular.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge(
                "contour", LegCall("extract_contours", ("main",))),
            DiagramEdge("polygon", LegCall("fit_polygon", ("contour",))),
            # This decorative witness node deliberately collides with the
            # unrelated score leg's name.
            DiagramEdge(
                "object_count", LegCall("classify_triangle", ("polygon",))),
            DiagramEdge("score", LegCall("object_count", ("scene",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=(MorphSpec("translate", "panel"),),
        semantic_requirements=("triangle",),
        witness_requirements=("TriangleWitness",),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert result.semantic_issue == "MISSING_LEG"
    assert "decorative" in result.compile_error


def test_problem_05_style_fish_proxy_is_not_semantic_clean():
    hyp = SemanticHypothesis(
        version="0.1",
        hypothesis_id="symmetric_fish_by_area",
        description="The positive figure is a symmetric fish-like object.",
            polarity="positive_satisfies",
            diagram=DiagramSpec((
                DiagramEdge("score", LegCall("total_ink", ("panel",))),
            )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("fish-like", "symmetric"),
    )
    result = verify_hypothesis(hyp, default_registry(), _problem_two_objects_vs_one())
    assert result.rule == "MISSING_LEG"
    assert result.missing_leg["semantic_term"] == "fish-like"


def test_two_intersecting_circles_requires_circle_pair_intersection_witness():
    hyp = SemanticHypothesis(
        version="0.1",
        hypothesis_id="two_circles_by_closure_proxy",
        description="The figure consists of two intersecting circles.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("score", LegCall("total_ink", ("panel",))),
        )),
        score_node="score",
        order="low_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("two circles", "intersect"),
    )
    result = verify_hypothesis(hyp, default_registry(), _problem_two_objects_vs_one())
    assert result.rule == "MISSING_LEG"
    assert result.missing_leg["semantic_term"] in {"circle", "two circles"}


def test_compiler_rejects_missing_leg():
    hyp = SemanticHypothesis(
        version="0.1",
        hypothesis_id="missing",
        description="A missing semantic relation.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("score", LegCall("not_in_registry", ("panel",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("missing semantic relation",),
    )
    try:
        compile_hypothesis(hyp, default_registry())
    except CompileError as exc:
        assert "missing semantic leg" in str(exc)
    else:
        raise AssertionError("missing leg should not compile")


def test_compile_failure_uses_panel_level_error_count():
    hyp = SemanticHypothesis(
        version="0.1",
        hypothesis_id="missing",
        description="A missing semantic relation.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("score", LegCall("not_in_registry", ("panel",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("missing semantic relation",),
    )
    result = verify_hypothesis(hyp, default_registry(), _problem_two_objects_vs_one())
    assert result.n_examples == 12
    assert result.support_errors == 12
    assert result.loo_errors == 12


def test_empty_gluing_cannot_launder_a_semantic_term():
    hyp = replace(
        _object_count_hypothesis(),
        hypothesis_id="triangle_via_empty_gluing",
        description="The figure is a triangle.",
        semantic_requirements=("triangle",),
        cofibrations=(CofibrationSpec(
            name="triangle",
            source_type="",
            target_type="",
            interface_fields=(),
            added_fields=(),
            attachment_leg="",
        ),),
    )
    result = verify_hypothesis(hyp, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert "missing required fields" in result.compile_error


def test_description_cannot_overclaim_structured_semantics():
    hyp = replace(
        _object_count_hypothesis(),
        hypothesis_id="bird_description_object_count_claim",
        description="The figure is bird-like.",
    )
    result = verify_hypothesis(hyp, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert "description names terms" in result.compile_error


def test_unknown_term_cannot_hide_beside_a_covered_term():
    hyp = replace(
        _object_count_hypothesis(),
        hypothesis_id="bird_connected_component_laundering",
        description="The figure is a bird-like connected component.",
        semantic_requirements=("bird-like connected component",),
    )
    result = verify_hypothesis(
        hyp, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert result.semantic_issue == "MISSING_LEG"
    assert "bird-like" in result.missing_leg["semantic_term"]
    assert "bird" in result.missing_leg["uncovered_tokens"]


def test_empty_or_stopword_only_semantic_declaration_is_not_a_gate():
    for requirement in (("",), ("the",)):
        hypothesis = replace(
            _object_count_hypothesis(),
            hypothesis_id="vacuous_semantic_declaration",
            semantic_requirements=requirement,
        )
        result = verify_hypothesis(
            hypothesis, default_registry(), _problem_two_objects_vs_one())
        assert not result.accepted
        assert result.compile_error


def test_proposal_cannot_control_complexity_or_silently_pass_parameters():
    complexity_gamed = replace(
        _object_count_hypothesis(), complexity_hint=-1000)
    result = verify_hypothesis(
        complexity_gamed, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert "complexity_hint" in result.compile_error

    base = _object_count_hypothesis()
    parameter_gamed = replace(
        base,
        diagram=DiagramSpec((
            base.diagram.edges[0],
            DiagramEdge(
                "score",
                LegCall("object_count", ("scene",), (("hidden", 1),)),
            ),
        )),
    )
    result = verify_hypothesis(
        parameter_gamed, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert "invalid call parameters" in result.compile_error


def test_unimplemented_contrast_is_rejected_instead_of_scored_zero_risk():
    hyp = replace(
        _object_count_hypothesis(),
        contrast_interventions=(MorphSpec(
            "remove_one_object", "panel", expected_effect="violate"),),
    )
    result = verify_hypothesis(hyp, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert "contrast_interventions are not executable" in result.compile_error


def test_runner_has_no_legacy_predicate_fallback():
    here = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(here, "run_semantic_cone.py"), encoding="utf-8").read()
    forbidden = ("bongard_api_agent", "bongard_legs", "bongard_arena", "predicates.py", "p_*")
    for marker in forbidden:
        assert marker not in text


def test_unrestricted_predicate_track_remains_available():
    import bongard_arena as A

    preds = {"p_ink": lambda panel: float(panel.sum())}
    result = A.verify(preds, _problem_two_objects_vs_one())
    assert result.rule.startswith("p_ink") or result.rule.startswith("CONST_")
    assert result.n_rotations == 36


def _part_fixture(part_id: str, src: str,
                  pts: tuple[tuple[float, float], ...]) -> PartWitness:
    return PartWitness(
        part_id=part_id, role="stroke", source_component_id=src,
        contour=ContourWitness(source_component_id=src, points=pts))


def _gluing_spec() -> CofibrationSpec:
    return CofibrationSpec(
        name="part_glued_into_graph",
        source_type="PartWitness",
        target_type="PartGraphWitness",
        interface_fields=("contacts",),
        added_fields=("parts",),
        attachment_leg="detect_attachment",
    )


def test_cofibration_is_a_gluing_not_an_inclusion():
    # IDs renamed, coordinates moved within tolerance: still a valid gluing.
    source = _part_fixture("body", "object_0", ((10.0, 10.0), (20.0, 20.0)))
    renamed = _part_fixture("part_7", "obj_A", ((10.6, 9.5), (20.4, 21.1)))
    other = _part_fixture("part_8", "obj_A", ((40.0, 40.0), (50.0, 50.0)))
    contact = ContactWitness(source_a="part_7", source_b="part_8",
                             points=(PointWitness(x=15.0, y=15.0),))
    target = PartGraphWitness(parts=(renamed, other), contacts=(contact,),
                              adjacency=(("part_7", "part_8"),))
    check = verify_cofibration(source, target, _gluing_spec())
    assert check.ok
    assert ("body", "part_7") in check.glue_map


def test_cofibration_gluing_rejects_broken_geometry_and_missing_interface():
    source = _part_fixture("body", "object_0", ((10.0, 10.0), (20.0, 20.0)))
    moved = _part_fixture("part_7", "obj_A", ((70.0, 70.0), (90.0, 90.0)))
    other = _part_fixture("part_8", "obj_A", ((40.0, 40.0), (50.0, 50.0)))
    contact = ContactWitness(source_a="part_7", source_b="part_8",
                             points=(PointWitness(x=15.0, y=15.0),))
    broken = PartGraphWitness(parts=(moved, other), contacts=(contact,),
                              adjacency=(("part_7", "part_8"),))
    failed = verify_cofibration(source, broken, _gluing_spec())
    assert not failed.ok
    assert failed.first_failed == "source_not_glued"

    matching = _part_fixture("part_7", "obj_A", ((10.0, 10.0), (20.0, 20.0)))
    no_interface = PartGraphWitness(parts=(matching, other), contacts=())
    failed2 = verify_cofibration(source, no_interface, _gluing_spec())
    assert not failed2.ok
    assert failed2.first_failed == "interface_missing"


def test_no_hardcoded_concept_gluings_in_library():
    here = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(here, "cofibrations.py"), encoding="utf-8").read()
    for marker in ("BIRD", "PINWHEEL", "TRIANGLE_SQUARE", "CIRCLE_INTERSECTION",
                   "bird", "pinwheel", "lamp", "fish"):
        assert marker not in text


def _panel_from_polylines(polylines) -> np.ndarray:
    grid = np.zeros((128, 128), dtype=np.uint8)
    for pts in polylines:
        _draw_polyline(grid, np.asarray(pts, dtype=float))
    return grid


def test_contour_closedness_is_honest():
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    circle = np.stack([64 + 30 * np.cos(theta), 64 + 30 * np.sin(theta)], axis=1)
    arc = circle[:120]
    closed_obj = L.select_largest(L.parse_scene(_panel_from_polylines([circle])))
    open_obj = L.select_largest(L.parse_scene(_panel_from_polylines([arc])))
    closed_contour = L.extract_contours(closed_obj)
    open_contour = L.extract_contours(open_obj)
    assert closed_contour.is_closed
    assert not open_contour.is_closed
    assert L.contour_closedness(closed_contour) == 1.0
    assert L.contour_closedness(open_contour) == 0.0
    assert L.fit_circle(closed_contour).residual < 0.1
    try:
        L.fit_circle(open_contour)
        raised = False
    except ValueError:
        raised = True
    assert raised  # an open arc is not a circle; the leg refuses honestly
    assert 90.0 <= L.fit_arc(open_contour).angle_degrees <= 260.0

    square = _panel_from_polylines([[
        (30, 30), (90, 30), (90, 90), (30, 90), (30, 30),
    ]])
    square_contour = L.extract_contours(
        L.select_largest(L.parse_scene(square)))
    try:
        L.fit_circle(square_contour)
        raised = False
    except L.WitnessAbsent:
        raised = True
    assert raised  # closed is necessary, not sufficient, for CircleWitness

    degenerate = ContourWitness(
        source_component_id="degenerate",
        points=((10.0, 10.0),) * 4,
        is_closed=True,
    )
    try:
        L.fit_circle(degenerate)
        failure = None
    except Exception as exc:
        failure = exc
    assert type(failure) is ValueError  # numerical failure is not semantic absence


def test_polygon_side_counts_from_strokes():
    tri = [(30, 30), (90, 30), (60, 80), (30, 30)]
    sq = [(30, 30), (90, 30), (90, 90), (30, 90), (30, 30)]
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    circle = np.stack([64 + 30 * np.cos(theta), 64 + 30 * np.sin(theta)], axis=1)

    def poly_of(polyline):
        scene = L.parse_scene(_panel_from_polylines([polyline]))
        return L.fit_polygon(L.extract_contours(L.select_largest(scene)))

    tri_poly, sq_poly = poly_of(tri), poly_of(sq)
    assert tri_poly.side_count == 3
    assert sq_poly.side_count == 4
    try:
        poly_of(circle)
        circle_failure = None
    except Exception as exc:
        circle_failure = exc
    assert type(circle_failure) is L.WitnessAbsent
    assert circle_failure.failure_mode == "too_few_sides"
    # A circle has no polygon witness; it is identified by a low circle-fit
    # residual rather than being treated as a many-sided polygon.
    circ_contour = L.extract_contours(L.select_largest(
        L.parse_scene(_panel_from_polylines([circle]))))
    assert L.fit_circle(circ_contour).residual < 0.1
    assert L.classify_triangle(tri_poly).confidence > 0.0
    assert L.classify_quadrilateral(sq_poly).confidence > 0.0
    for bad, cls in ((sq_poly, L.classify_triangle),):
        try:
            cls(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised


def test_contact_and_intersection_witnesses_are_honest():
    cross = _panel_from_polylines([[(64, 20), (64, 108)], [(20, 64), (108, 64)]])
    tee = _panel_from_polylines([[(30, 30), (90, 30)], [(60, 30), (60, 90)]])
    apart = _panel_from_polylines([[(20, 20), (40, 20)], [(80, 80), (100, 80)]])

    g_cross = L.build_part_graph(L.parse_scene(cross))
    assert L.part_count(g_cross) >= 3
    assert L.intersection_count(g_cross) >= 1
    assert L.detect_intersection(g_cross).relation == "intersection"
    try:
        L.detect_attachment(g_cross)
        attachment_failure = None
    except Exception as exc:
        attachment_failure = exc
    assert type(attachment_failure) is L.WitnessAbsent
    assert attachment_failure.failure_mode == "no_attachment"

    g_tee = L.build_part_graph(L.parse_scene(tee))
    assert L.contact_count(g_tee) >= 1
    assert L.detect_attachment(g_tee).relation == "attachment"
    try:
        L.detect_intersection(g_tee)
        raised = False
    except ValueError:
        raised = True
    assert raised  # a T-junction is attachment, not a crossing

    g_apart = L.build_part_graph(L.parse_scene(apart))
    assert L.contact_count(g_apart) == 0.0
    try:
        L.detect_contact(g_apart)
        raised = False
    except ValueError:
        raised = True
    assert raised  # no fabricated centroid-midpoint contact


def test_skeleton_topology_counts_junctions_without_inventing_loops():
    fixtures = {
        "line": (_panel_from_polylines([[(20, 64), (108, 64)]]), (2, 0, 0)),
        "tee": (_panel_from_polylines([
            [(20, 40), (108, 40)], [(64, 40), (64, 100)],
        ]), (3, 1, 0)),
        "cross": (_panel_from_polylines([
            [(20, 64), (108, 64)], [(64, 20), (64, 108)],
        ]), (4, 1, 0)),
        "loop": (_panel_from_polylines([[
            (30, 30), (98, 30), (98, 98), (30, 98), (30, 30),
        ]]), (0, 0, 1)),
    }
    for panel, expected in fixtures.values():
        graph = L.build_skeleton_graph(
            L.select_largest(L.parse_scene(panel)))
        assert (graph.endpoint_count, graph.branch_count,
                graph.cycle_count) == expected
        assert graph.nodes
        assert graph.edges
        assert all(0 <= left < len(graph.nodes)
                   and 0 <= right < len(graph.nodes)
                   for left, right in graph.edges)


def test_absolute_count_and_direction_words_are_executable_semantics():
    def count_panel(component_count: int, offset: int) -> np.ndarray:
        panel = np.zeros((128, 128), dtype=np.uint8)
        for index in range(component_count):
            y = 18 + index * 32 + offset
            x = 20 + index * 29
            panel[y:y + 10, x:x + 10] = 1
        return panel

    def intersection_panel(count: int, offset: int) -> np.ndarray:
        centers = ((38 + offset, 34), (90 + offset, 88))[:count]
        lines = []
        for cx, cy in centers:
            lines.extend((
                [(cx - 15, cy), (cx + 15, cy)],
                [(cx, cy - 15), (cx, cy + 15)],
            ))
        return _panel_from_polylines(lines)

    def intersection_hypothesis(term: str) -> SemanticHypothesis:
        return SemanticHypothesis(
            version="0.1",
            hypothesis_id=f"calibrated_{term.replace(' ', '_')}",
            description=f"Positive figures have {term}.",
            polarity="positive_satisfies",
            diagram=DiagramSpec((
                DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
                DiagramEdge(
                    "graph", LegCall("build_part_graph", ("scene",))),
                DiagramEdge(
                    "score", LegCall("intersection_count", ("graph",))),
            )),
            score_node="score",
            order="low_positive",
            preservation_morphisms=_morphism(),
            semantic_requirements=(term,),
        )

    false_absence = Problem(
        "relative_is_not_absence", "fixture", "harness_only",
        tuple(intersection_panel(1, offset) for offset in range(-3, 3)),
        tuple(intersection_panel(2, offset) for offset in range(-3, 3)),
    )
    result = verify_hypothesis(
        intersection_hypothesis("no intersection"),
        default_registry(), false_absence)
    assert not result.accepted
    # Absolute absence is the decision rule itself, not a post-hoc annotation
    # on an unrelated learned threshold.
    assert result.support_errors == 6
    assert result.semantic_issue == "semantic_count_positive_violates_exact"

    exact_one = SemanticHypothesis(
        version="0.1",
        hypothesis_id="exactly_one_component",
        description="Positive panels have one component.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("score", LegCall("object_count", ("scene",))),
        )),
        score_node="score",
        order="low_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("one component",),
    )
    false_cardinal = Problem(
        "relative_is_not_one", "fixture", "harness_only",
        tuple(count_panel(2, offset) for offset in range(6)),
        tuple(count_panel(3, offset) for offset in range(6)),
    )
    result = verify_hypothesis(
        exact_one, default_registry(), false_cardinal)
    assert not result.accepted
    assert result.support_errors == 6
    assert result.semantic_issue == "semantic_count_positive_violates_exact"

    squares = tuple(_panel_from_polylines([[
        (30 + offset, 30), (90 + offset, 30), (90 + offset, 90),
        (30 + offset, 90), (30 + offset, 30),
    ]]) for offset in range(-3, 3))
    lines = tuple(_panel_from_polylines([[
        (25 + offset, 60), (100 + offset, 60),
    ]]) for offset in range(-3, 3))
    open_hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id="open_curve_direction",
        description="The positive object is an open curve.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge(
                "contour", LegCall("extract_contours", ("main",))),
            DiagramEdge(
                "score", LegCall("contour_closedness", ("contour",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("open curve",),
    )
    inverted = verify_hypothesis(
        open_hypothesis, default_registry(),
        Problem("inverted_open", "fixture", "harness_only", squares, lines))
    assert not inverted.accepted
    assert inverted.support_errors == 0
    assert inverted.semantic_issue.startswith(
        "semantic_score_direction_mismatch:low")

    honest_absence = verify_hypothesis(
        intersection_hypothesis("no intersection"), default_registry(),
        Problem(
            "honest_absence", "fixture", "harness_only",
            tuple(_panel_from_polylines([
                [(18 + offset, 24), (48 + offset, 24)],
                [(78 + offset, 82), (108 + offset, 82)],
            ]) for offset in range(-3, 3)),
            tuple(intersection_panel(1, offset) for offset in range(-3, 3)),
        ))
    assert honest_absence.accepted

    honest_cardinal = verify_hypothesis(
        exact_one, default_registry(),
        Problem(
            "honest_one", "fixture", "harness_only",
            tuple(count_panel(1, offset) for offset in range(6)),
            tuple(count_panel(2, offset) for offset in range(6)),
        ))
    assert honest_cardinal.accepted

    honest_open = verify_hypothesis(
        replace(open_hypothesis, order="low_positive"), default_registry(),
        Problem("honest_open", "fixture", "harness_only", lines, squares))
    assert honest_open.accepted


def test_description_cannot_hide_score_calibration_operators():
    hypothesis = replace(
        _object_count_hypothesis(),
        description="Positive panels have no components.",
        semantic_requirements=("component",),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert "disagree on score calibration" in result.compile_error


def test_witness_presence_cannot_be_replaced_by_relative_fit_confidence():
    skinny = tuple(_panel_from_polylines([[
        (30 + offset, 30), (98 + offset, 30),
        (38 + offset, 90), (30 + offset, 30),
    ]]) for offset in range(-3, 3))
    regular = tuple(_panel_from_polylines([[
        (30 + offset, 30), (98 + offset, 30),
        (64 + offset, 90), (30 + offset, 30),
    ]]) for offset in range(-3, 3))
    problem = Problem(
        "triangles_on_both_sides", "fixture", "harness_only",
        skinny, regular)

    categorical = verify_hypothesis(
        _triangle_witness_hypothesis(), default_registry(), problem)
    # The fixed witness-presence predicate honestly counts every negative
    # triangle as a support error.
    assert categorical.support_errors == 6
    assert not categorical.accepted
    assert categorical.semantic_issue \
        == "semantic_witness_claim_present_on_negative"

    explicitly_relative = replace(
        _triangle_witness_hypothesis(),
        hypothesis_id="higher_triangle_fit_confidence",
        description="Positive triangles have higher confidence.",
        semantic_requirements=("triangle", "higher confidence"),
    )
    relative = verify_hypothesis(
        explicitly_relative, default_registry(), problem)
    assert relative.accepted


def test_absolute_magnitude_word_is_not_an_arbitrary_relative_threshold():
    def rectangle(width: int, offset: int) -> np.ndarray:
        panel = np.zeros((96, 96), dtype=np.uint8)
        panel[20 + offset:70 + offset, 20:20 + width] = 1
        return panel

    problem = Problem(
        "near_squares_are_not_thin", "fixture", "harness_only",
        tuple(rectangle(49, offset) for offset in range(6)),
        tuple(rectangle(50, offset) for offset in range(6)),
    )
    hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id="absolute_high_aspect",
        description="Positive objects have a high aspect ratio.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge("score", LegCall("bbox_aspect", ("main",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("high aspect ratio",),
    )
    result = verify_hypothesis(hypothesis, default_registry(), problem)
    assert result.support_errors == 0
    assert not result.accepted
    assert result.semantic_issue \
        == "semantic_relative_measurement_requires_direction"


def test_circle_pair_intersection_constructs_real_points():
    def circle(source_id, x, y, radius):
        return L.CircleWitness(
            source_component_id=source_id,
            center=PointWitness(x=x, y=y, source_id=source_id),
            radius=radius,
        )

    def pair(first, second, distance):
        return L.CirclePairWitness(
            first=first,
            second=second,
            center_distance=distance,
        )

    def assert_on_both(point, first, second):
        for candidate in (first, second):
            distance = np.hypot(
                point.x - candidate.center.x,
                point.y - candidate.center.y,
            )
            assert abs(distance - candidate.radius) < 1e-9

    first = circle("a", 0.0, 0.0, 5.0)
    second = circle("b", 6.0, 0.0, 5.0)
    intersection = L.circle_pair_intersection(pair(first, second, 6.0))
    assert intersection.relation == "intersection"
    assert len(intersection.points) == 2
    assert {(round(point.x, 9), round(point.y, 9))
            for point in intersection.points} == {(3.0, -4.0), (3.0, 4.0)}
    for point in intersection.points:
        assert_on_both(point, first, second)
        assert (point.x, point.y) != (3.0, 0.0)  # the old midpoint proxy

    tangent = circle("tangent", 10.0, 0.0, 5.0)
    tangent_intersection = L.circle_pair_intersection(
        pair(first, tangent, 10.0))
    assert len(tangent_intersection.points) == 1
    assert_on_both(tangent_intersection.points[0], first, tangent)
    assert (tangent_intersection.points[0].x,
            tangent_intersection.points[0].y) == (5.0, 0.0)


def test_intersecting_raster_circles_are_reachable_end_to_end():
    theta = np.linspace(0.0, 2.0 * np.pi, 360)
    panel = _panel_from_polylines([
        np.stack((48.0 + 28.0 * np.cos(theta),
                  64.0 + 28.0 * np.sin(theta)), axis=1),
        np.stack((80.0 + 28.0 * np.cos(theta),
                  64.0 + 28.0 * np.sin(theta)), axis=1),
    ])
    scene = L.parse_scene(panel)
    assert len(scene.objects) == 1  # intersecting strokes really are merged

    pair = L.fit_multiple_circles(scene)
    assert pair.first.source_component_id != pair.second.source_component_id
    assert sorted((pair.first.radius, pair.second.radius)) == pytest.approx(
        (28.0, 28.0), abs=0.2)
    intersection = L.circle_pair_intersection(pair)
    assert len(intersection.points) == 2
    for point in intersection.points:
        for circle in (pair.first, pair.second):
            assert math.hypot(
                point.x - circle.center.x,
                point.y - circle.center.y,
            ) == pytest.approx(circle.radius, abs=1e-9)

    shifted = np.zeros_like(panel)
    shifted[7:, 11:] = panel[:-7, :-11]
    shifted_pair = L.fit_multiple_circles(L.parse_scene(shifted))
    before = sorted((circle.center.x, circle.center.y, circle.radius)
                    for circle in (pair.first, pair.second))
    after = sorted((circle.center.x, circle.center.y, circle.radius)
                   for circle in (shifted_pair.first, shifted_pair.second))
    assert after == pytest.approx(
        [(x + 11.0, y + 7.0, radius) for x, y, radius in before],
        abs=0.12,
    )


def test_circle_pair_intersection_refuses_absent_or_malformed_geometry():
    def circle(source_id, x, y, radius):
        return L.CircleWitness(
            source_component_id=source_id,
            center=PointWitness(x=x, y=y, source_id=source_id),
            radius=radius,
        )

    def pair(first, second, distance):
        return L.CirclePairWitness(
            first=first,
            second=second,
            center_distance=distance,
        )

    first = circle("a", 0.0, 0.0, 5.0)
    absent_pairs = (
        pair(first, circle("external", 11.0, 0.0, 5.0), 11.0),
        pair(first, circle("contained", 1.0, 0.0, 2.0), 1.0),
        pair(first, circle("coincident", 0.0, 0.0, 5.0), 0.0),
    )
    for absent_pair in absent_pairs:
        try:
            L.circle_pair_intersection(absent_pair)
            failure = None
        except Exception as exc:
            failure = exc
        assert type(failure) is L.WitnessAbsent

    malformed_pairs = (
        pair(first, circle("wrong-distance", 6.0, 0.0, 5.0), 7.0),
        pair(first, circle("zero-radius", 6.0, 0.0, 0.0), 6.0),
        pair(first, circle("nan-center", float("nan"), 0.0, 5.0),
             float("nan")),
    )
    for malformed_pair in malformed_pairs:
        try:
            L.circle_pair_intersection(malformed_pair)
            failure = None
        except Exception as exc:
            failure = exc
        assert type(failure) is ValueError


def test_compiled_gluing_binds_projection_attachment_and_positive_panels():
    positives = []
    negatives = []
    for offset in range(-3, 3):
        positives.append(_panel_from_polylines([
            [(30 + offset, 30), (90 + offset, 30)],
            [(60 + offset, 30), (60 + offset, 90)],
        ]))
        negatives.append(_panel_from_polylines([
            [(18 + offset, 24), (48 + offset, 24)],
            [(78 + offset, 82), (108 + offset, 82)],
        ]))
    problem = Problem(
        "attachment_fixture", "fixture", "attached_parts",
        tuple(positives), tuple(negatives))
    hypothesis = _attachment_hypothesis()
    result = verify_hypothesis(hypothesis, default_registry(), problem)
    assert result.accepted
    assert result.support_errors == 0
    assert result.rotated_loo_errors == 0
    assert result.cofibration_errors == 0
    assert result.structural_absences == 6


def test_crossing_cannot_discharge_an_attachment_gluing():
    crosses = []
    separated = []
    for offset in range(-3, 3):
        crosses.append(_panel_from_polylines([
            [(20 + offset, 64), (108 + offset, 64)],
            [(64 + offset, 20), (64 + offset, 108)],
        ]))
        separated.append(_panel_from_polylines([
            [(18 + offset, 24), (48 + offset, 24)],
            [(78 + offset, 82), (108 + offset, 82)],
        ]))
    problem = Problem(
        "cross_is_not_attachment", "fixture", "harness_only",
        tuple(crosses), tuple(separated))
    result = verify_hypothesis(
        _attachment_hypothesis(), default_registry(), problem)
    assert not result.accepted
    assert result.support_errors == 6
    assert result.cofibration_errors == 6
    assert result.witness_absences == {
        "pos:attachment:detect_attachment:no_attachment": 6,
        "neg:attachment:detect_attachment:no_attachment": 6,
    }


def test_gluing_interface_and_patch_fields_must_be_disjoint():
    source = _part_fixture(
        "body", "object_0", ((10.0, 10.0), (20.0, 20.0)))
    other = _part_fixture(
        "other", "object_0", ((30.0, 10.0), (40.0, 20.0)))
    contact = ContactWitness(
        source_a="body", source_b="other",
        points=(PointWitness(x=25.0, y=15.0),), relation="attachment")
    target = PartGraphWitness(
        parts=(source, other), contacts=(contact,),
        adjacency=(("body", "other"),))
    degenerate = replace(
        _gluing_spec(), interface_fields=("parts",), added_fields=("parts",))
    # The low-level predicate demonstrates why the compiler's static guard is
    # needed: overlapping prose would otherwise prove no patch addition.
    assert verify_cofibration(source, target, degenerate).ok

    base = _object_count_hypothesis()
    hypothesis = replace(
        base,
        hypothesis_id="overlapping_gluing_fields",
        cofibrations=(CofibrationSpec(
            name="overlap",
            source_node="scene",
            target_node="score",
            source_type="Scene",
            target_type="Measurement",
            interface_fields=("value",),
            added_fields=("value",),
            attachment_leg="object_count",
        ),),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert "must be disjoint" in result.compile_error


def test_gluing_cannot_use_witness_bookkeeping_as_patch_structure():
    hypothesis = replace(
        _attachment_hypothesis(),
        hypothesis_id="bookkeeping_is_not_gluing_structure",
        cofibrations=(replace(
            _attachment_hypothesis().cofibrations[0],
            interface_fields=("confidence",),
            added_fields=("residual",),
        ),),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert "witness bookkeeping" in result.compile_error


def test_expected_witness_absence_is_a_negative_decision_not_a_crash():
    triangles = []
    squares = []
    for offset in range(-3, 3):
        triangles.append(_panel_from_polylines([[
            (30 + offset, 30), (90 + offset, 30),
            (60 + offset, 80), (30 + offset, 30),
        ]]))
        squares.append(_panel_from_polylines([[
            (30 + offset, 30), (90 + offset, 30),
            (90 + offset, 90), (30 + offset, 90),
            (30 + offset, 30),
        ]]))
    problem = Problem(
        "triangle_fixture", "fixture", "triangle",
        tuple(triangles), tuple(squares))
    hyp = replace(
        _triangle_witness_hypothesis(),
        preservation_morphisms=(MorphSpec("translate", "panel"),),
    )
    result = verify_hypothesis(hyp, default_registry(), problem)
    assert result.accepted
    assert result.support_errors == 0
    assert result.loo_errors == 0
    assert result.predicate_errors == 0
    assert result.structural_absences == 6
    assert result.witness_absences == {
        "neg:triangle:classify_triangle:wrong_side_count": 6,
    }
    assert result.scores[6:] == (None,) * 6
    assert result.score_dispositions == \
        ("present",) * 6 + ("semantic_absent",) * 6
    assert result.support_predictions == (True,) * 6 + (False,) * 6


def test_empty_scene_selector_is_absent_instead_of_fabricating_an_object():
    empty = np.zeros((96, 96), dtype=np.uint8)
    negatives = []
    for offset in range(6):
        panel = np.zeros_like(empty)
        panel[24 + offset:54 + offset, 30:60] = 1
        negatives.append(panel)
    problem = Problem(
        "empty_selector_fixture", "fixture", "harness_only",
        (empty.copy(),) * 6, tuple(negatives))
    hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id="empty_is_not_narrow",
        description="Positive objects have a higher aspect ratio.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge("score", LegCall("bbox_aspect", ("main",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=(MorphSpec("translate", "panel"),),
        semantic_requirements=("higher aspect ratio",),
    )
    result = verify_hypothesis(hypothesis, default_registry(), problem)
    assert not result.accepted
    assert result.support_errors >= 6
    assert result.structural_absences == 6
    assert result.witness_absences == {
        "pos:main:select_largest:no_objects": 6,
    }


def test_runtime_leg_contracts_reject_wrong_codomains_and_undeclared_absence():
    def registry_with(contract):
        registry = LegRegistry()
        for existing in default_registry().contracts():
            registry.register(existing)
        registry.register(contract)
        return registry

    def hypothesis(leg_name):
        return SemanticHypothesis(
            version="0.1",
            hypothesis_id=f"contract_{leg_name}",
            description=f"The panel has a {leg_name} measure.",
            polarity="positive_satisfies",
            diagram=DiagramSpec((
                DiagramEdge("score", LegCall(leg_name, ("panel",))),
            )),
            score_node="score",
            order="high_positive",
            preservation_morphisms=_morphism(),
            semantic_requirements=(leg_name,),
        )

    wrong_type = LegContract(
        name="wrong_measure",
        domain=("Panel",),
        codomain="Measurement",
        implementation=lambda panel: ContourWitness(points=((1.0, 1.0),)),
        measurement_kind="continuous",
    )
    cone = compile_hypothesis(
        hypothesis(wrong_type.name), registry_with(wrong_type))
    trace = cone.trace(np.zeros((16, 16), dtype=np.uint8),
                       registry_with(wrong_type))
    assert trace.leg_status["score"] == "error:TypeError"
    assert "codomain contract violation" in trace.errors[0]

    def raises_undeclared(_panel):
        raise WitnessAbsent("invented_absence", "not in the contract")

    undeclared = LegContract(
        name="undeclared_absence",
        domain=("Panel",),
        codomain="Measurement",
        implementation=raises_undeclared,
        measurement_kind="continuous",
    )
    registry = registry_with(undeclared)
    cone = compile_hypothesis(hypothesis(undeclared.name), registry)
    trace = cone.trace(np.zeros((16, 16), dtype=np.uint8), registry)
    assert trace.leg_status["score"] == "error:UndeclaredWitnessAbsence"
    assert "invented_absence" in trace.errors[0]

    fractional_count = LegContract(
        name="fractional_count",
        domain=("Panel",),
        codomain="Measurement",
        implementation=lambda panel: 1.5,
        measurement_kind="count",
    )
    registry = registry_with(fractional_count)
    cone = compile_hypothesis(hypothesis(fractional_count.name), registry)
    trace = cone.trace(np.zeros((16, 16), dtype=np.uint8), registry)
    assert trace.leg_status["score"] == "error:TypeError"
    assert "nonnegative integer count" in trace.errors[0]

    missing_kind = LegContract(
        name="uncalibrated_measurement",
        domain=("Panel",),
        codomain="Measurement",
        implementation=lambda panel: 1.0,
    )
    registry = LegRegistry()
    try:
        registry.register(missing_kind)
        missing_kind_failure = None
    except Exception as exc:
        missing_kind_failure = exc
    assert type(missing_kind_failure) is ValueError
    assert "requires one of" in str(missing_kind_failure)


def test_declared_morphisms_are_executed_or_reported_unchecked():
    hypothesis = replace(
        _object_count_hypothesis(),
        preservation_morphisms=(
            MorphSpec("translate", "panel"),
            MorphSpec("uniform_scale", "panel"),
        ),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(),
        _problem_two_objects_vs_one())
    assert not result.accepted
    assert result.declared_morphism_checks == 12
    assert result.unchecked_morphisms == ("uniform_scale",)


def test_ignored_morphism_parameters_and_unrelated_gluings_are_rejected():
    parameterized = replace(
        _object_count_hypothesis(),
        preservation_morphisms=(MorphSpec(
            "translate", "panel", parameters={"pixels": 99}),),
    )
    result = verify_hypothesis(
        parameterized, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert "parameters are not executable" in result.compile_error

    unrelated = replace(
        _object_count_hypothesis(),
        hypothesis_id="unrelated_gluing",
        cofibrations=(CofibrationSpec(
            name="object_count_gluing",
            source_node="scene",
            target_node="score",
            source_type="Scene",
            target_type="Measurement",
            interface_fields=("objects",),
            added_fields=("value",),
            attachment_leg="contour_closedness",
        ),),
    )
    result = verify_hypothesis(
        unrelated, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert "must accept declared target type" in result.compile_error


def test_placeholder_semantic_contracts_are_not_proposer_visible():
    names = set(default_registry().names())
    quarantined = {
        "binarize_panel",
        "bbox_fill",
        "build_containment_tree",
        "closure_ratio",
        "decompose_curve_into_arcs_and_lines",
        "decompose_into_line_segments",
        "estimate_curvature",
        "estimate_tangents",
        "curvature_extrema",
        "detect_tangency",
        "detect_shared_endpoint",
        "detect_shared_point",
        "reflection_symmetry",
        "rotational_symmetry_order",
        "pair_parts_by_symmetry",
        "select_inner_object",
        "select_outer_object",
        "largest_area",
        "symmetry_residual",
        "symmetry_order_score",
    }
    assert names.isdisjoint(quarantined)
    assert "elongation" in names


def test_elongation_contract_survives_exact_nuisance_actions():
    base = np.zeros((128, 128), dtype=np.uint8)
    base[52:60, 35:75] = 1

    translated = np.zeros_like(base)
    translated[63:71, 47:87] = 1
    scaled_crop = np.repeat(
        np.repeat(base[45:67, 25:85], 2, axis=0), 2, axis=1)
    scaled = np.zeros_like(base)
    scaled[:scaled_crop.shape[0], :scaled_crop.shape[1]] = scaled_crop

    def score(panel):
        return L.elongation(L.select_largest(L.parse_scene(panel)))

    reference = score(base)
    assert score(np.rot90(base)) == reference
    assert score(np.fliplr(base)) == reference
    assert score(translated) == reference
    assert abs(score(scaled) - reference) / reference < 0.01


def test_structured_proposal_parsing_is_per_item_tolerant():
    good = {
        "hypothesis_id": "h1",
        "description": "The principal object is an open curve.",
        "polarity": "positive_satisfies",
        "diagram": {"edges": [
            {"target": "scene", "call": {"leg_name": "parse_scene", "args": ["panel"]}},
            {"target": "main", "call": {"leg_name": "select_largest", "args": ["scene"]}},
            {"target": "contour", "call": {"leg_name": "extract_contours", "args": ["main"]}},
            {"target": "score", "call": {"leg_name": "contour_closedness", "args": ["contour"]}},
        ]},
        "score_node": "score",
        "order": "low_positive",
        "semantic_requirements": ["open curve"],
        "witness_requirements": ["ContourWitness"],
        "preservation_morphisms": [{
            "name": "translate", "scope": "panel",
            "expected_effect": "preserve",
        }],
    }
    bad = {"description": "no hypothesis_id or score_node"}
    hyps, err = hypotheses_from_tool_input(
        {"hypotheses": [good, bad, bad]})
    assert len(hyps) == 1
    assert hyps[0].hypothesis_id == "h1"
    assert "hypothesis[1]" in err
    result = verify_hypothesis(hyps[0], default_registry(),
                               _problem_two_objects_vs_one())
    assert result.compile_error == ""

    good2 = dict(good)
    good2["hypothesis_id"] = "h2"
    duplicate, duplicate_err = hypotheses_from_tool_input(
        {"hypotheses": [good, good, good2]})
    assert len(duplicate) == 2
    assert "duplicate hypothesis_id" in duplicate_err

    for invalid_count in (1, 9):
        proposals, count_err = hypotheses_from_tool_input(
            {"hypotheses": [good] * invalid_count})
        assert proposals == ()
        assert "expected between 3 and 8" in count_err


def test_runner_reserves_solved_status_for_every_exact_gate():
    verification = verify_hypothesis(
        _object_count_hypothesis(), default_registry(),
        _problem_two_objects_vs_one())
    bundle = ProposalBundle(
        "fixture", (_object_count_hypothesis(),), "", "fixture")
    assert _status_of(verification, bundle) == "SOLVED_SEMANTIC_PURE"
    approximate = replace(
        verification, accepted=True, rotated_loo_errors=1)
    assert _status_of(approximate, bundle) == "APPROXIMATE_SEMANTIC_FIT"
    unchecked = replace(
        verification, accepted=False,
        unchecked_morphisms=("uniform_scale",))
    assert _status_of(unchecked, bundle) == "MORPHISM_UNCHECKED"


def test_live_selector_records_conditional_risk_without_zeroing_unknowns():
    verification = verify_hypothesis(
        _object_count_hypothesis(), default_registry(),
        _problem_two_objects_vs_one())
    assert verification.semantic_admissible
    assert verification.risk.R_support == 0.0
    assert verification.risk.R_rotated_LOO == 0.0
    assert verification.risk.R_naturality == 0.0
    assert verification.risk.R_contrast is None
    assert verification.complexity == verification.complexity_breakdown.total

    selected = _select([verification], lambda_value=0.02)
    assert selected is verification
    record = _selection_record(
        selected, [verification], lambda_value=0.02)
    assert record["conditional_free_energy"] >= 0.0
    assert record["free_energy"] is None
    assert "R_contrast" in record["unmeasured_risks"]

    failed = verify_hypothesis(
        _triangle_proxy_hypothesis(), default_registry(),
        _problem_two_objects_vs_one())
    assert _select([failed]) is failed
    failed_record = _selection_record(failed, [failed], lambda_value=0.02)
    assert failed_record["conditional_free_energy"] is None
    assert "R_support" in failed_record["conditional_unmeasured_risks"]


def test_live_selector_cannot_trade_exact_acceptance_for_lower_complexity():
    base = verify_hypothesis(
        _object_count_hypothesis(), default_registry(),
        _problem_two_objects_vs_one())
    exact_breakdown = replace(
        base.complexity_breakdown, diagram_node_cost=100)
    exact = replace(
        base, hypothesis_id="exact_but_expensive",
        complexity=exact_breakdown.total,
        complexity_breakdown=exact_breakdown)
    approximate = replace(
        base,
        hypothesis_id="cheap_policy_reject",
        accepted=False,
        semantic_admissible=True,
        support_errors=1,
        loo_errors=1,
        rotated_loo_errors=6,
        complexity=1,
        complexity_breakdown=ComplexityBreakdown(diagram_node_cost=1),
        risk=RiskVector(
            R_support=1 / 12,
            R_rotated_LOO=6 / 72,
            R_naturality=0.0,
            R_parser_stability=0.0,
        ),
    )
    assert _select([approximate, exact]) is exact


def test_runner_feedback_uses_fixed_rule_predictions_not_rule_string_parsing():
    verification = verify_hypothesis(
        _operator_count_hypothesis("two components", "high_positive"),
        default_registry(),
        _operator_count_problem((2,) * 6, (1, 3, 1, 3, 1, 3)),
    )
    assert verification.accepted
    assert verification.rule == "score==2"
    assert _misses(verification) == ""


def test_prompt_leg_list_is_generated_from_registry():
    prompt = build_prompt("problem_00")
    for name in default_registry().names():
        assert f"- {name}:" in prompt
    # no black-box composite concept legs are advertised
    for forbidden in ("bird", "fish", "lamp", "pinwheel", "prototype"):
        assert forbidden not in prompt.lower()


def test_terminal_replay_preserves_missing_leg_as_revalidated_evidence():
    problem, record = _terminal_candidate_fixture(
        _triangle_proxy_hypothesis())
    assert record.status == "MISSING_LEG"
    replayed = _replay_terminal_record(
        record, problem,
        max_support_errors=0,
        max_loo_errors=0,
        max_rotated_loo_errors=0,
        lambda_value=0.02,
    )
    assert replayed["summary"]["status"] == "MISSING_LEG"
    assert replayed["selected_verification"]["semantic_issue"] == \
        "MISSING_LEG"
    assert replayed["selected_verification"]["missing_leg"]


@pytest.mark.parametrize(("field", "forged"), (
    ("solved", True),
    ("status", "NO_PROPOSALS"),
    ("selected_hypothesis", "forged"),
    ("selected_description", "forged"),
    ("selected_rule", "forged"),
    ("support_errors", -1),
    ("loo_errors", -1),
    ("rotated_loo_errors", -1),
    ("rotated_loo_checks", -1),
    ("n_examples", -1),
    ("complexity", -1),
    ("rounds_used", 2),
    ("proposer_kind", "forged"),
    ("proposer_error", "forged"),
))
def test_terminal_replay_rejects_forged_checkpoint_summary(field, forged):
    problem, record = _terminal_candidate_fixture(
        _triangle_proxy_hypothesis())
    tampered = asdict(record)
    tampered[field] = forged
    with pytest.raises(ValueError, match="does not replay"):
        _replay_terminal_record(
            tampered, problem,
            max_support_errors=0,
            max_loo_errors=0,
            max_rotated_loo_errors=0,
            lambda_value=0.02,
        )


def test_terminal_replay_rejects_impossible_proposer_candidate_count():
    problem, record = _terminal_candidate_fixture(
        _triangle_proxy_hypothesis())
    tampered = asdict(record)
    round_record = tampered["terminal_evidence"]["rounds"][0]
    round_record["candidate_count"] = 9
    round_record["candidate_ids"] = [f"candidate-{index}" for index in range(9)]
    round_record["hypothesis_digests"] = [
        "sha256:" + f"{index:064x}" for index in range(9)]
    tampered["terminal_evidence_digest"] = SR.canonical_json_digest(
        tampered["terminal_evidence"])
    with pytest.raises(ValueError, match="candidate bound"):
        _replay_terminal_record(
            tampered, problem,
            max_support_errors=0,
            max_loo_errors=0,
            max_rotated_loo_errors=0,
            lambda_value=0.02,
            round_limit=4,
        )


def test_terminal_replay_rejects_impossible_four_receipt_round():
    problem, record = _terminal_candidate_fixture(
        _triangle_proxy_hypothesis())
    tampered = asdict(record)
    tampered["terminal_evidence"]["rounds"][0]["model_receipts"] = [
        {"receipt": index} for index in range(4)]
    tampered["terminal_evidence_digest"] = SR.canonical_json_digest(
        tampered["terminal_evidence"])
    with pytest.raises(ValueError, match="receipt bound"):
        _replay_terminal_record(
            tampered, problem,
            max_support_errors=0,
            max_loo_errors=0,
            max_rotated_loo_errors=0,
            lambda_value=0.02,
            round_limit=4,
        )


def test_terminal_replay_rejects_resealed_candidate_verification_tamper():
    problem, record = _terminal_candidate_fixture(
        _triangle_proxy_hypothesis())
    tampered = asdict(record)
    expected = tampered["terminal_evidence"]["selection"]["candidates"][0][
        "expected_verification"]
    expected["support_errors"] += 1
    tampered["candidates"][0]["support_errors"] += 1
    tampered["support_errors"] += 1
    tampered["terminal_evidence_digest"] = SR.canonical_json_digest(
        tampered["terminal_evidence"])
    with pytest.raises(ValueError, match="verification does not replay"):
        _replay_terminal_record(
            tampered, problem,
            max_support_errors=0,
            max_loo_errors=0,
            max_rotated_loo_errors=0,
            lambda_value=0.02,
        )


def test_parse_failed_empty_terminal_has_explicit_replayable_evidence():
    source = _problem_two_objects_vs_one()
    problem = Problem(
        source.problem_id, "basic", source.concept, source.pos, source.neg)
    rounds = [{
        "round": 0,
        "proposer_kind": "offline-test",
        "parse_error": "invalid proposal schema",
        "candidate_count": 0,
        "candidate_ids": [],
        "hypothesis_digests": [],
    }]
    evidence = _terminal_evidence(rounds, [], [], None, 0.02, [])
    n_examples = len(problem.pos) + len(problem.neg)
    record = ProblemResult(
        opaque_id="problem_00", category="basic", solved=False,
        selected_hypothesis="", selected_description="", selected_rule="",
        support_errors=n_examples, loo_errors=n_examples,
        rotated_loo_errors=72, rotated_loo_checks=72,
        n_examples=n_examples, complexity=0, rounds_used=1,
        proposer_kind="offline-test", track="SEMANTIC-PURE",
        condition=PD.OBSERVED, sharing_policy=PD.SHARED,
        corpus_digest="sha256:" + "1" * 64,
        panel_set_digest=SR.panel_set_digest(
            SR.panel_records_from_problem(problem)),
        control_digest="", status="PROPOSER_PARSE_FAILED",
        proposer_error="invalid proposal schema", candidates=[],
        candidate_manifest=[], selection={}, terminal_evidence=evidence,
        terminal_evidence_digest=SR.canonical_json_digest(evidence),
    )
    replayed = _replay_terminal_record(
        record, problem,
        max_support_errors=0,
        max_loo_errors=0,
        max_rotated_loo_errors=0,
        lambda_value=0.02,
    )
    assert replayed["summary"]["status"] == "PROPOSER_PARSE_FAILED"

    tampered = asdict(record)
    tampered["terminal_evidence"]["proposal_outcome"] = "NO_PROPOSALS"
    tampered["terminal_evidence_digest"] = SR.canonical_json_digest(
        tampered["terminal_evidence"])
    with pytest.raises(ValueError, match="proposal outcome"):
        _replay_terminal_record(
            tampered, problem,
            max_support_errors=0,
            max_loo_errors=0,
            max_rotated_loo_errors=0,
            lambda_value=0.02,
        )


def _terminal_resume_fixture(tmp_path):
    problem, record = _terminal_candidate_fixture(
        _triangle_proxy_hypothesis())
    manifest = PD.build_corpus_manifest(
        [problem], source="basic", seed=17, limit_per_source=1,
        dataset_revision="unavailable")
    bundle = PD.build_corpus_bundle([problem], manifest)
    record = replace(
        record,
        corpus_digest=manifest["corpus_digest"],
        panel_set_digest=manifest["problems"][0]["panel_set_digest"],
    )
    args = SimpleNamespace(
        condition=PD.OBSERVED,
        proposer="anthropic",
        model="offline",
        max_tokens=1,
        rounds=1,
        max_support_errors=0,
        max_loo_errors=0,
        max_rotated_loo_errors=0,
        lambda_value=0.02,
        source="basic",
        seed=17,
        limit=1,
        tag="offline",
    )
    payload = _checkpoint_payload(
        args, [record], manifest, 1, None, bundle)
    SA.atomic_json(os.path.join(tmp_path, "checkpoint.json"), payload)
    return problem, record, manifest, bundle, args, payload


def test_resume_replays_failed_terminal_record_and_rejects_summary_tamper(
        tmp_path):
    problem, record, manifest, bundle, args, payload = \
        _terminal_resume_fixture(tmp_path)
    resumed, _results, promoted = _load_resume_state(
        str(tmp_path), args, manifest, None, 1, [problem], bundle)
    assert SR.canonical_json_digest([asdict(item) for item in resumed]) == \
        SR.canonical_json_digest([asdict(record)])
    assert promoted == []

    tampered = copy.deepcopy(payload)
    tampered["records"][0]["selected_description"] = "forged"
    SA.atomic_json(os.path.join(tmp_path, "checkpoint.json"), tampered)
    with pytest.raises(SystemExit, match="terminal record problem_00"):
        _load_resume_state(
            str(tmp_path), args, manifest, None, 1, [problem], bundle)


def test_solved_resume_binds_full_terminal_evidence_to_runspec(
        tmp_path, monkeypatch):
    problem, record = _terminal_candidate_fixture()
    manifest = PD.build_corpus_manifest(
        [problem], source="basic", seed=17, limit_per_source=1,
        dataset_revision="unavailable")
    bundle = PD.build_corpus_bundle([problem], manifest)
    args = SimpleNamespace(
        condition=PD.OBSERVED,
        proposer="anthropic",
        model="offline",
        max_tokens=1,
        rounds=1,
        max_support_errors=0,
        max_loo_errors=0,
        max_rotated_loo_errors=0,
        lambda_value=0.02,
        source="basic",
        seed=17,
        limit=1,
        tag="offline",
    )
    hypothesis = _object_count_hypothesis()
    verification = verify_hypothesis(
        hypothesis, default_registry(), problem)
    origins = [{
        "round": 0,
        "round_candidate_index": 0,
        "round_candidate_count": 1,
    }]
    monkeypatch.setattr(SR, "BONGARD_ROOT", tmp_path)
    spec = _write_replay_spec(
        args, str(tmp_path), "problem_00", problem,
        hypothesis.to_dict(), verification, default_registry(),
        [verification], [hypothesis.to_dict()], origins,
        manifest, manifest["problems"][0], None, None,
        bundle["bundle_digest"], record.terminal_evidence,
    )
    record = replace(
        record,
        corpus_digest=manifest["corpus_digest"],
        panel_set_digest=manifest["problems"][0]["panel_set_digest"],
        replay_spec_digest=spec.spec_digest,
    )
    payload = _checkpoint_payload(
        args, [record], manifest, 1, None, bundle)
    SA.atomic_json(os.path.join(tmp_path, "checkpoint.json"), payload)
    resumed, _results, promoted = _load_resume_state(
        str(tmp_path), args, manifest, None, 1, [problem], bundle)
    assert resumed[0].replay_spec_digest == spec.spec_digest
    assert SR.semantic_cone_digest(promoted[0]["hypothesis"]) == \
        SR.semantic_cone_digest(hypothesis.to_dict())

    spec_path = os.path.join(tmp_path, "replay_specs", "problem_00.json")
    proposer_tamper = json.loads(open(spec_path, encoding="utf-8").read())
    proposer_tamper["provenance"]["proposer"]["model"] = "forged-model"
    proposer_tamper["spec_digest"] = SR.canonical_json_digest({
        key: value for key, value in proposer_tamper.items()
        if key != "spec_digest"})
    SA.atomic_json(spec_path, proposer_tamper)
    proposer_checkpoint = copy.deepcopy(payload)
    proposer_checkpoint["records"][0]["replay_spec_digest"] = \
        proposer_tamper["spec_digest"]
    SA.atomic_json(os.path.join(tmp_path, "checkpoint.json"), proposer_checkpoint)
    with pytest.raises(SystemExit, match="differs from its RunSpec"):
        _load_resume_state(
            str(tmp_path), args, manifest, None, 1, [problem], bundle)
    SR.save_runspec(spec_path, spec, allowed_root=tmp_path)

    tampered = copy.deepcopy(payload)
    tampered["records"][0]["terminal_evidence"]["rounds"][0][
        "proposer_kind"] = "forged"
    tampered["records"][0]["proposer_kind"] = "forged"
    tampered["records"][0]["terminal_evidence_digest"] = \
        SR.canonical_json_digest(
            tampered["records"][0]["terminal_evidence"])
    SA.atomic_json(os.path.join(tmp_path, "checkpoint.json"), tampered)
    with pytest.raises(SystemExit, match="differs from its RunSpec"):
        _load_resume_state(
            str(tmp_path), args, manifest, None, 1, [problem], bundle)


def test_conflicting_checkpoint_preflight_leaves_all_bindings_byte_identical(
        tmp_path, monkeypatch):
    source = _problem_two_objects_vs_one()
    problem = Problem(
        source.problem_id, "basic", source.concept, source.pos, source.neg)
    monkeypatch.setattr(
        PD, "sample_corpus", lambda *args, **kwargs: [problem])
    monkeypatch.setattr(SR, "BONGARD_ROOT", tmp_path)
    out_dir = tmp_path / "preflight_run"
    observed_args = SimpleNamespace(
        proposer="anthropic",
        out_dir=str(out_dir),
        max_support_errors=0,
        max_loo_errors=0,
        max_rotated_loo_errors=0,
        limit=1,
        rounds=1,
        max_tokens=1,
        corpus_size=1,
        lambda_value=0.02,
        dataset_dir=str(tmp_path / "dataset"),
        seed=17,
        source="basic",
        condition=PD.OBSERVED,
        control_seed=91,
        control_replicate=0,
        prepare_only=True,
        model="offline",
        tag="preflight",
        preregistration="",
        arm_id="",
    )
    import run_semantic_cone as runner
    runner.run(observed_args)
    manifest = json.loads(
        (out_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    bundle = json.loads(
        (out_dir / "corpus_panels.json").read_text(encoding="utf-8"))
    SA.atomic_json(
        str(out_dir / "checkpoint.json"),
        _checkpoint_payload(
            observed_args, [], manifest, 1, None, bundle),
    )

    def tree_snapshot():
        entries = {}
        for root, dirnames, filenames in os.walk(out_dir):
            relative_root = os.path.relpath(root, out_dir)
            entries[("dir", relative_root)] = tuple(sorted(dirnames))
            for filename in sorted(filenames):
                path = os.path.join(root, filename)
                entries[("file", os.path.relpath(path, out_dir))] = \
                    open(path, "rb").read()
        return entries

    before = tree_snapshot()
    shuffled_args = SimpleNamespace(**{
        **vars(observed_args),
        "condition": PD.SHUFFLED_SIDES,
        "prepare_only": False,
    })
    monkeypatch.setattr(
        runner, "AnthropicCofiberedProposer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("proposer constructed before conflict preflight")),
    )
    with pytest.raises(SystemExit, match="active-prefix/run policy"):
        runner.run(shuffled_args)
    assert tree_snapshot() == before
    assert not (out_dir / "control_manifest.json").exists()
    assert not (out_dir / "workspace").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
@pytest.mark.parametrize("kind", ("symlink", "hardlink", "fifo", "oversize"))
def test_semantic_preflight_json_rejects_unsafe_existing_files(
        tmp_path, kind):
    import run_semantic_cone as runner

    destination = tmp_path / "corpus_panels.json"
    if kind == "symlink":
        source = tmp_path / "symlink-target.json"
        source.write_text("{}\n", encoding="utf-8")
        destination.symlink_to(source)
    elif kind == "hardlink":
        source = tmp_path / "second-name.json"
        source.write_text("{}\n", encoding="utf-8")
        os.link(source, destination)
    elif kind == "fifo":
        os.mkfifo(destination)
    else:
        with open(destination, "wb") as handle:
            handle.truncate(runner.artifact_io.MAX_JSON_BYTES + 1)

    with pytest.raises(SystemExit, match="existing.*invalid"):
        runner._read_preflight_json(str(destination), "corpus panel bundle")


def test_semantic_preflight_json_rejects_path_replacement_during_read(
        tmp_path, monkeypatch):
    import run_semantic_cone as runner

    destination = tmp_path / "control_manifest.json"
    payload = {"stable": True}
    encoded = SR.canonical_json_bytes(payload) + b"\n"
    destination.write_bytes(encoded)
    replacement = tmp_path / "replacement.json"
    original_lstat = os.lstat
    replaced = False

    def replace_before_identity_check(path, *args, **kwargs):
        nonlocal replaced
        if os.path.abspath(os.fspath(path)) == os.path.abspath(destination) \
                and not replaced:
            replaced = True
            replacement.write_bytes(encoded)
            os.replace(replacement, destination)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(
        runner.artifact_io.os, "lstat", replace_before_identity_check)
    with pytest.raises(SystemExit, match="changed while being read"):
        runner._read_preflight_json(str(destination), "control manifest")
    assert replaced


@pytest.mark.parametrize("binding_kind", ("manifest", "bundle", "control"))
def test_semantic_bindings_preserve_a_concurrent_create_once_winner(
        tmp_path, monkeypatch, binding_kind):
    import run_semantic_cone as runner

    source_problem = _problem_two_objects_vs_one()
    problem = Problem(
        source_problem.problem_id, "basic", source_problem.concept,
        source_problem.pos, source_problem.neg)
    manifest = PD.build_corpus_manifest(
        [problem], source="basic", seed=17, limit_per_source=1,
        dataset_revision="unavailable")
    bundle = PD.build_corpus_bundle([problem], manifest)
    control = PD.build_shuffled_sides_control(
        [problem], manifest, seed=73, replicate=0).manifest
    out_dir = tmp_path / "create_once"
    out_dir.mkdir()

    if binding_kind == "manifest":
        changed_panel = problem.pos[0].copy()
        changed_panel[0, 0] ^= 1
        other_problem = Problem(
            problem.problem_id, problem.category, problem.concept,
            (changed_panel,) + problem.pos[1:], problem.neg)
        winner = PD.build_corpus_manifest(
            [other_problem], source="basic", seed=17, limit_per_source=1,
            dataset_revision="unavailable")
        destination = out_dir / "corpus_manifest.json"
        invoke = lambda: runner._bind_corpus_manifest(
            str(out_dir), manifest)
        message = "different corpus"
    elif binding_kind == "bundle":
        # No different bundle can validate against the same manifest; an
        # invalid concurrent winner must still be retained and rejected.
        winner = {"concurrent": "winner"}
        destination = out_dir / "corpus_panels.json"
        invoke = lambda: runner._bind_corpus_bundle(
            str(out_dir), bundle, manifest)
        message = "invalid"
    else:
        winner = PD.build_shuffled_sides_control(
            [problem], manifest, seed=74, replicate=0).manifest
        destination = out_dir / "control_manifest.json"
        invoke = lambda: runner._bind_control_manifest(
            str(out_dir), control, manifest)
        message = "different control"

    called = False

    def concurrent_winner(path, payload):
        nonlocal called
        assert os.path.abspath(path) == os.path.abspath(destination)
        assert not called
        called = True
        destination.write_bytes(SR.canonical_json_bytes(winner) + b"\n")
        return False

    monkeypatch.setattr(SA, "create_json_once", concurrent_winner)
    with pytest.raises(SystemExit, match=message):
        invoke()
    assert called
    assert runner.artifact_io._load_json(
        str(destination), "concurrent winner") == winner


def test_run_report_replays_failed_record_instead_of_trusting_matching_summaries(
        tmp_path):
    problem, record, manifest, bundle, _args, payload = \
        _terminal_resume_fixture(tmp_path)
    results = {"problem_00": _result_payload(problem, record)}
    validated = SA._validate_run_inputs(
        payload, results, manifest, bundle, None, require_complete=True)
    assert validated["terminal_replays"]["problem_00"]["summary"][
        "status"] == "MISSING_LEG"

    forged_payload = copy.deepcopy(payload)
    forged_results = copy.deepcopy(results)
    forged_payload["records"][0]["status"] = "NO_PROPOSALS"
    forged_results["problem_00"]["status"] = "NO_PROPOSALS"
    with pytest.raises(SA.ReplayCertificationError,
                       match="terminal record problem_00"):
        SA._validate_run_inputs(
            forged_payload, forged_results, manifest, bundle, None,
            require_complete=True)

    forged_results = copy.deepcopy(results)
    forged_results["problem_00"]["support_errors"] += 1
    with pytest.raises(SA.ReplayCertificationError,
                       match="support_errors"):
        SA._validate_run_inputs(
            payload, forged_results, manifest, bundle, None,
            require_complete=True)

    extra_summary = copy.deepcopy(payload)
    extra_summary["records"][0]["claimed_accuracy"] = 1.0
    with pytest.raises(SA.ReplayCertificationError,
                       match="terminal schema"):
        SA._validate_run_inputs(
            extra_summary, results, manifest, bundle, None,
            require_complete=True)


@pytest.mark.parametrize("identity_error", ("tag", "out_dir"))
def test_preregistered_execution_identity_conflicts_fail_before_any_write(
        tmp_path, monkeypatch, identity_error):
    source_problem = _problem_two_objects_vs_one()
    problem = Problem(
        source_problem.problem_id, "basic", source_problem.concept,
        source_problem.pos, source_problem.neg)
    import run_semantic_cone as runner
    monkeypatch.setattr(
        runner.phase_d_protocol, "sample_corpus",
        lambda *args, **kwargs: [problem])
    corpus_manifest = PD.build_corpus_manifest(
        [problem], source="basic", seed=17, limit_per_source=1,
        dataset_revision="unavailable", dataset_inputs_digest="unavailable")
    preregistration = PD.build_preregistration(
        corpus_manifest,
        tracks=["SEMANTIC-PURE"],
        scales=[1],
        shuffled_seed=73,
        shuffled_replicates=1,
    )
    arm = next(
        item for item in preregistration["arms"]
        if item["arm_id"] == "SEMANTIC-PURE:primary:n1")
    binding = PD.execution_binding(preregistration, arm["arm_id"])
    preregistration_path = tmp_path / "preregistration.json"
    preregistration_path.write_text(
        json.dumps(preregistration), encoding="utf-8")

    monkeypatch.setattr(runner.semantic_replay, "BONGARD_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "AnthropicCofiberedProposer",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("proposer constructed before identity preflight")),
    )

    args = SimpleNamespace(
        proposer="anthropic",
        out_dir=str(tmp_path / "wrong_out_dir"),
        max_support_errors=0,
        max_loo_errors=0,
        max_rotated_loo_errors=0,
        limit=1,
        rounds=4,
        max_tokens=8000,
        corpus_size=1,
        lambda_value=0.02,
        dataset_dir=str(tmp_path / "dataset"),
        seed=17,
        source="basic",
        condition=PD.OBSERVED,
        control_seed=73,
        control_replicate=0,
        prepare_only=False,
        model="sonnet",
        tag=arm["execution_tag"],
        preregistration=str(preregistration_path),
        arm_id=arm["arm_id"],
    )
    expected_message = "canonical arm path"
    if identity_error == "tag":
        args.tag = "wrong-execution-tag"
        expected_message = "arm.execution_tag"

    def snapshot():
        return {
            str(path.relative_to(tmp_path)): (
                "DIR" if path.is_dir() else path.read_bytes())
            for path in sorted(tmp_path.rglob("*"))
        }

    before = snapshot()
    with pytest.raises(SystemExit, match=expected_message):
        runner.run(args)
    assert snapshot() == before
    assert not (tmp_path / "wrong_out_dir").exists()


def _nested_semantic_growth_context(tmp_path, monkeypatch, *, start=True):
    source = _problem_two_objects_vs_one()
    problems = [
        Problem(
            f"fixture_{index}", "basic", f"concept_{index}",
            source.pos, source.neg)
        for index in range(5)
    ]
    import run_semantic_cone as runner
    monkeypatch.setattr(
        runner.phase_d_protocol, "sample_corpus",
        lambda *args, **kwargs: problems)
    monkeypatch.setattr(runner.semantic_replay, "BONGARD_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "SEMANTIC_RUNS_DIR", str(tmp_path / "semantic_runs"))
    monkeypatch.setattr(
        SA, "artifact_dir",
        lambda tag: str(tmp_path / "artifacts" / f"{tag}_semantic"))

    manifest = PD.build_corpus_manifest(
        problems, source="basic", seed=17, limit_per_source=5,
        dataset_revision="unavailable", dataset_inputs_digest="unavailable")
    preregistration = PD.build_preregistration(
        manifest,
        tracks=["SEMANTIC-PURE"],
        scales=[1, 5],
        shuffled_seed=73,
        shuffled_replicates=1,
    )
    arms = {
        arm["scale"]: arm for arm in preregistration["arms"]
        if arm["track"] == "SEMANTIC-PURE"
        and arm["condition"] == "primary"
    }
    assert arms[1]["execution_tag"] == arms[5]["execution_tag"]
    preregistration_path = tmp_path / "preregistration.json"
    preregistration_path.write_text(
        json.dumps(preregistration), encoding="utf-8")

    receipt = {
        "schema": PD.SEMANTIC_PROPOSER_RECEIPT_SCHEMA,
        "source": "anthropic-messages-api",
        "requested_model": "claude-sonnet-5",
        "actual_model": "claude-sonnet-5",
        "input_tokens": 10,
        "output_tokens": 2,
        "stop_reason": "tool_use",
    }
    receipt["receipt_digest"] = SR.canonical_json_digest(receipt)

    class EmptyProposer:
        constructions = 0

        def __init__(self, model, max_tokens):
            type(self).constructions += 1

        @staticmethod
        def _empty(problem_id):
            return ProposalBundle(
                problem_id=problem_id,
                hypotheses=(),
                raw_text="offline empty proposal",
                proposer_kind="anthropic",
                model_receipts=(receipt,),
            )

        def propose(self, problem_id, panel_paths):
            assert len(panel_paths) == 12
            return self._empty(problem_id)

        def refine(self, problem_id, feedback):
            return self._empty(problem_id)

    monkeypatch.setattr(runner, "AnthropicCofiberedProposer", EmptyProposer)
    execution_tag = arms[1]["execution_tag"]
    out_dir = os.path.join(
        runner.SEMANTIC_RUNS_DIR, execution_tag)

    def arguments(scale):
        arm = arms[scale]
        return SimpleNamespace(
            proposer="anthropic",
            out_dir=out_dir,
            max_support_errors=0,
            max_loo_errors=0,
            max_rotated_loo_errors=0,
            limit=5,
            rounds=4,
            max_tokens=8000,
            corpus_size=scale,
            lambda_value=0.02,
            dataset_dir=str(tmp_path / "dataset"),
            seed=17,
            source="basic",
            condition=PD.OBSERVED,
            control_seed=73,
            control_replicate=0,
            prepare_only=False,
            model="sonnet",
            tag=execution_tag,
            preregistration=str(preregistration_path),
            arm_id=arm["arm_id"],
        )

    if start:
        runner.run(arguments(1))
    return runner, EmptyProposer, arguments, out_dir


def test_preregistered_family_rejects_fresh_n5_before_any_write(
        tmp_path, monkeypatch):
    runner, proposer, arguments, out_dir = \
        _nested_semantic_growth_context(tmp_path, monkeypatch, start=False)

    def snapshot():
        return {
            str(path.relative_to(tmp_path)): (
                "DIR" if path.is_dir() else path.read_bytes())
            for path in sorted(tmp_path.rglob("*"))
        }

    before = snapshot()
    constructions = proposer.constructions
    with pytest.raises(SystemExit, match="immediate predecessor"):
        runner.run(arguments(5))
    assert snapshot() == before
    assert proposer.constructions == constructions
    assert not os.path.exists(out_dir)


def test_preregistered_family_checkpoint_grows_from_n1_to_n5(
        tmp_path, monkeypatch):
    runner, _proposer, arguments, out_dir = \
        _nested_semantic_growth_context(tmp_path, monkeypatch)
    checkpoint_n1 = json.loads(open(
        os.path.join(out_dir, "checkpoint.json"), encoding="utf-8").read())
    assert checkpoint_n1["dataset"]["active_prefix_size"] == 1
    assert checkpoint_n1["attempted"] == 1

    runner.run(arguments(5))
    checkpoint_n5 = json.loads(open(
        os.path.join(out_dir, "checkpoint.json"), encoding="utf-8").read())
    assert checkpoint_n5["dataset"]["active_prefix_size"] == 5
    assert checkpoint_n5["attempted"] == 5
    assert SR.canonical_json_digest(checkpoint_n5["records"][:1]) == \
        SR.canonical_json_digest(checkpoint_n1["records"])


def test_preregistered_family_checkpoint_rejects_n5_to_n1_without_writes(
        tmp_path, monkeypatch):
    runner, proposer, arguments, out_dir = \
        _nested_semantic_growth_context(tmp_path, monkeypatch)
    runner.run(arguments(5))

    def snapshot():
        return {
            str(path.relative_to(tmp_path)): (
                "DIR" if path.is_dir() else path.read_bytes())
            for path in sorted(tmp_path.rglob("*"))
        }

    before = snapshot()
    constructions = proposer.constructions
    with pytest.raises(SystemExit, match="active prefix"):
        runner.run(arguments(1))
    assert snapshot() == before
    assert proposer.constructions == constructions


def test_anthropic_proposer_rejects_provider_model_substitution():
    proposer = AnthropicCofiberedProposer("sonnet", 8000)
    proposer._require_response_model(SimpleNamespace(model="claude-sonnet-5"))
    with pytest.raises(RuntimeError, match="omitted.*provider model"):
        proposer._require_response_model(SimpleNamespace())
    with pytest.raises(RuntimeError, match="differs.*requested concrete"):
        proposer._require_response_model(
            SimpleNamespace(model="claude-opus-4-8"))


def test_preregistered_growth_rejects_incomplete_smaller_prefix_without_writes(
        tmp_path, monkeypatch):
    runner, proposer, arguments, out_dir = \
        _nested_semantic_growth_context(tmp_path, monkeypatch)
    checkpoint_path = os.path.join(out_dir, "checkpoint.json")
    checkpoint = json.loads(open(
        checkpoint_path, encoding="utf-8").read())
    checkpoint["attempted"] = 0
    SA.atomic_json(checkpoint_path, checkpoint)

    def snapshot():
        return {
            str(path.relative_to(tmp_path)): (
                "DIR" if path.is_dir() else path.read_bytes())
            for path in sorted(tmp_path.rglob("*"))
        }

    before = snapshot()
    constructions = proposer.constructions
    with pytest.raises(SystemExit, match="completed terminal prefix"):
        runner.run(arguments(5))
    assert snapshot() == before
    assert proposer.constructions == constructions


def test_solved_finalization_crash_retries_without_destructive_run_report(
        tmp_path, monkeypatch):
    import run_semantic_cone as runner

    retained = tmp_path / "previous_runspec.json"
    retained.write_text("retained", encoding="utf-8")
    calls = []

    def forbidden_publish(*args, **kwargs):
        retained.unlink()
        calls.append("publish")
        return "published"

    def crashing_promote(*args, **kwargs):
        calls.append("promote-crash")
        raise RuntimeError("injected promotion crash")

    monkeypatch.setattr(SA, "publish_run_report", forbidden_publish)
    monkeypatch.setattr(SA, "promote", crashing_promote)
    arguments = SimpleNamespace(tag="phase-semantic")
    call = lambda: runner._finalize_semantic_artifact(
        arguments, str(tmp_path), {"attempted": 1}, {},
        [{"opaque_id": "problem_00"}], {}, {}, None)
    with pytest.raises(RuntimeError, match="injected promotion crash"):
        call()
    assert retained.read_text(encoding="utf-8") == "retained"
    assert calls == ["promote-crash"]

    monkeypatch.setattr(
        SA, "promote",
        lambda *args, **kwargs: calls.append("promote-retry") or "resumed")
    assert call() == "resumed"
    assert retained.read_text(encoding="utf-8") == "retained"
    assert calls == ["promote-crash", "promote-retry"]


def test_semantic_phase_d_publisher_emits_reproducible_track_report(
        tmp_path, monkeypatch):
    problem = _problem_two_objects_vs_one()
    manifest_problem = Problem(
        problem.problem_id, "basic", problem.concept,
        problem.pos, problem.neg)
    corpus_manifest = PD.build_corpus_manifest(
        [manifest_problem],
        source="basic",
        seed=17,
        limit_per_source=1,
        dataset_revision="unavailable",
    )
    preregistration = PD.build_preregistration(
        corpus_manifest,
        tracks=["SEMANTIC-PURE"],
        scales=[1],
        shuffled_seed=73,
        shuffled_replicates=1,
    )
    arm = next(
        item for item in preregistration["arms"]
        if item["arm_id"] == "SEMANTIC-PURE:primary:n1")
    binding = PD.execution_binding(preregistration, arm["arm_id"])
    receipt = {
        "schema": PD.SEMANTIC_PROPOSER_RECEIPT_SCHEMA,
        "source": "anthropic-messages-api",
        "requested_model": "claude-sonnet-5",
        "actual_model": "claude-sonnet-5",
        "input_tokens": 10,
        "output_tokens": 2,
        "stop_reason": "tool_use",
    }
    receipt["receipt_digest"] = SR.canonical_json_digest(receipt)
    terminal_evidence = _terminal_evidence([{
        "round": 0,
        "proposer_kind": "anthropic",
        "parse_error": "",
        "candidate_count": 0,
        "candidate_ids": [],
        "hypothesis_digests": [],
        "model_receipts": [receipt],
    }], [], [], None, 0.02, [])
    record = ProblemResult(
        opaque_id="problem_00",
        category="basic",
        solved=False,
        selected_hypothesis="",
        selected_description="",
        selected_rule="",
        support_errors=12,
        loo_errors=12,
        rotated_loo_errors=72,
        rotated_loo_checks=72,
        n_examples=12,
        complexity=0,
        rounds_used=1,
        proposer_kind="anthropic",
        track="SEMANTIC-PURE",
        condition=PD.OBSERVED,
        sharing_policy=PD.SHARED,
        corpus_digest=corpus_manifest["corpus_digest"],
        panel_set_digest=corpus_manifest["problems"][0][
            "panel_set_digest"],
        control_digest="",
        status="NO_PROPOSALS",
        proposer_error="",
        candidates=[],
        candidate_manifest=[],
        terminal_evidence=terminal_evidence,
        terminal_evidence_digest=SR.canonical_json_digest(terminal_evidence),
        phase_execution_binding_digest=binding["binding_digest"],
    )
    monkeypatch.setattr(
        SA, "artifact_dir", lambda tag: str(tmp_path / f"{tag}_semantic"))

    with pytest.raises(SystemExit, match="arm.execution_tag"):
        _publish_phase_d_track_report(
            "publisher_test", preregistration, arm, [record])
    assert not list(tmp_path.iterdir())

    path = _publish_phase_d_track_report(
        arm["execution_tag"], preregistration, arm, [record])
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)

    PD.validate_track_report(report, preregistration)
    raw_trace = SR.canonical_json_digest([asdict(record)])
    assert report["report_source_trace_digest"] == raw_trace
    assert report["records"][0]["report_source_trace_digest"] == raw_trace
    assert PD._report_source_trace_digest(
        report["track"], report["records"]) == raw_trace

    report_path = os.fspath(path)
    backing = tmp_path / "track-report-backing.json"
    os.replace(report_path, backing)
    os.link(backing, report_path)
    with pytest.raises(SystemExit, match="existing semantic track report"):
        _publish_phase_d_track_report(
            arm["execution_tag"], preregistration, arm, [record])


def test_semantic_artifact_taint_and_promotion():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "run")
        os.makedirs(out)
        with open(os.path.join(out, "problem_00_round00_proposal.txt"), "w",
                  encoding="utf-8") as f:
            f.write("clean typed proposal")
        with open(os.path.join(out, "checkpoint.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        assert SA.taint_reason(out) is None
        bad = os.path.join(out, "notes.txt")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("peeked at get_action_string_list")
        assert SA.taint_reason(out) is not None
        os.remove(bad)

        problem = _problem_two_objects_vs_one()
        hypothesis = _object_count_hypothesis()
        verification = verify_hypothesis(
            hypothesis, default_registry(), problem)
        loser_hypothesis = replace(
            hypothesis, hypothesis_id="zz_equivalent_loser")
        loser_verification = verify_hypothesis(
            loser_hypothesis, default_registry(), problem)
        candidates = [verification, loser_verification]
        candidate_hypotheses = [
            hypothesis.to_dict(), loser_hypothesis.to_dict()]
        assert _select(candidates, 0.02) is verification
        selection_evidence = _selection_evidence(
            candidates, candidate_hypotheses, verification, 0.02)
        assert {item["round_candidate_count"] for item in
                selection_evidence["candidate_manifest"]} == {2}
        with pytest.raises(ValueError, match="indices are incomplete"):
            _selection_evidence(
                candidates, candidate_hypotheses, verification, 0.02,
                [{
                    "round": 0,
                    "round_candidate_index": index,
                    "round_candidate_count": 3,
                } for index in range(2)],
            )
        selection_record = _selection_record(
            verification, candidates, 0.02)
        candidate_origins = [{
            "round": 0,
            "round_candidate_index": index,
            "round_candidate_count": len(candidates),
        } for index in range(len(candidates))]
        round_trace = [{
            "round": 0,
            "proposer_kind": "offline-test",
            "parse_error": "",
            "candidate_count": len(candidates),
            "candidate_ids": [
                item.hypothesis_id for item in
                (hypothesis, loser_hypothesis)],
            "hypothesis_digests": [
                SR.semantic_cone_digest(item.to_dict()) for item in
                (hypothesis, loser_hypothesis)],
        }]
        terminal_evidence = _terminal_evidence(
            round_trace, candidates, candidate_hypotheses,
            verification, 0.02, candidate_origins)
        panel_set_digest = SR.panel_set_digest(
            SR.panel_records_from_problem(problem))
        manifest_problem = Problem(
            problem.problem_id, "basic", problem.concept,
            problem.pos, problem.neg)
        corpus_manifest = PD.build_corpus_manifest(
            [manifest_problem],
            source="basic",
            seed=1,
            limit_per_source=1,
            dataset_revision="unavailable",
        )
        corpus_digest = corpus_manifest["corpus_digest"]
        corpus_bundle = PD.build_corpus_bundle(
            [manifest_problem], corpus_manifest)
        SA.atomic_json(
            os.path.join(out, "corpus_manifest.json"), corpus_manifest)
        SA.atomic_json(
            os.path.join(out, "corpus_panels.json"), corpus_bundle)

        def protocol_provenance(selection):
            bound_terminal = copy.deepcopy(terminal_evidence)
            bound_terminal["selection"] = selection
            return {
                "runner": "unit_test",
                "dataset": {
                    "source": corpus_manifest["sampling"]["source"],
                    "seed": corpus_manifest["sampling"]["seed"],
                    "limit_per_source": corpus_manifest["sampling"][
                        "limit_per_source"],
                    "count_policy": PD.COUNT_POLICY,
                    "order_policy": corpus_manifest["sampling"][
                        "order_policy"],
                    "repository_commit": corpus_manifest["sampling"][
                        "dataset_revision"],
                    "corpus_digest": corpus_digest,
                    "corpus_bundle_digest": corpus_bundle["bundle_digest"],
                    "panel_set_digest": panel_set_digest,
                    "panels": "self-contained; source identifier redacted",
                },
                "experiment": {
                    "track": "SEMANTIC-PURE",
                    "condition": "observed",
                    "sharing_policy": "shared",
                    "control": None,
                },
                "proposer": {
                    "kind": "offline-test",
                    "model": "offline",
                    "round_limit": 1,
                },
                "selection": selection,
                "terminal": {
                    "schema": bound_terminal["schema"],
                    "proposal_outcome": bound_terminal[
                        "proposal_outcome"],
                    "rounds": bound_terminal["rounds"],
                    "evidence_digest": SR.canonical_json_digest(
                        bound_terminal),
                },
            }

        spec = SR.build_runspec(
            opaque_id="problem_00",
            problem=problem,
            cones=[hypothesis],
            registry=default_registry(),
            verifier=verify_hypothesis,
            policy=SR.VerifierPolicy(unexecuted_checks=(
                "contrast", "counterfactual", "archive_regression")),
            expected_verifications={
                hypothesis.hypothesis_id: verification.to_dict()},
            provenance=protocol_provenance(selection_evidence),
            verifier_sources=verifier_related_sources(),
        )
        SR.save_runspec(
            os.path.join(out, "replay_specs", "problem_00.json"),
            spec,
            allowed_root=td,
            create_parents=True,
        )
        promoted = [{
            "opaque_id": "problem_00",
            "hypothesis": hypothesis.to_dict(),
            "verification": verification.to_dict(),
            "selection": selection_record,
            "runspec_digest": spec.spec_digest,
            "rounds_used": 1,
        }]

        def checkpoint_for(active_spec, *,
                           recorded_candidates=None,
                           recorded_selection=None):
            checkpoint_record = ProblemResult(
                opaque_id="problem_00",
                category="basic",
                solved=True,
                selected_hypothesis=hypothesis.hypothesis_id,
                selected_description=hypothesis.description,
                selected_rule=verification.rule,
                support_errors=verification.support_errors,
                loo_errors=verification.loo_errors,
                rotated_loo_errors=verification.rotated_loo_errors,
                rotated_loo_checks=verification.rotated_loo_checks,
                n_examples=verification.n_examples,
                complexity=verification.complexity,
                rounds_used=1,
                proposer_kind="offline-test",
                track="SEMANTIC-PURE",
                condition="observed",
                sharing_policy="shared",
                corpus_digest=corpus_digest,
                panel_set_digest=panel_set_digest,
                control_digest="",
                status="SOLVED_SEMANTIC_PURE",
                proposer_error="",
                candidates=[candidate.to_dict() for candidate in candidates],
                candidate_manifest=selection_evidence["candidate_manifest"],
                selection=selection_record,
                terminal_evidence=terminal_evidence,
                terminal_evidence_digest=SR.canonical_json_digest(
                    terminal_evidence),
                replay_spec_digest=active_spec.spec_digest,
            )
            checkpoint_record = asdict(checkpoint_record)
            if recorded_candidates is not None:
                checkpoint_record["candidates"] = recorded_candidates
            if recorded_selection is not None:
                checkpoint_record["selection"] = recorded_selection
            return {
                "condition": "observed",
                "sharing_policy": "shared",
                "control": None,
                "dataset": {
                    "source": corpus_manifest["sampling"]["source"],
                    "seed": corpus_manifest["sampling"]["seed"],
                    "count_policy": PD.COUNT_POLICY,
                    "limit_per_source": corpus_manifest["sampling"][
                        "limit_per_source"],
                    "corpus_digest": corpus_digest,
                    "corpus_bundle_digest": corpus_bundle["bundle_digest"],
                    "active_prefix_size": 1,
                    "frozen_problem_count": 1,
                    "order_policy": corpus_manifest["sampling"][
                        "order_policy"],
                    "repository_commit": corpus_manifest["sampling"][
                        "dataset_revision"],
                    "corpus_manifest": "corpus_manifest.json",
                    "corpus_bundle": "corpus_panels.json",
                    "panel_bytes": (
                        "all records bind corpus panel-set digests; solved "
                        "replay_specs also embed canonical panel bytes"),
                },
                "solved": 1,
                "attempted": 1,
                "verifier_policy": {
                    "max_support_errors": 0,
                    "max_threshold_loo_errors": 0,
                    "max_pair_threshold_loo_errors": 0,
                },
                "selection": {"lambda": 0.02},
                "proposer": "offline-test",
                "model": "offline",
                "rounds": 1,
                "records": [checkpoint_record],
            }

        with pytest.raises(SA.ReplayCertificationError, match="runspec_digest"):
            SA._cold_replay_specs(out, [{
                **promoted[0], "runspec_digest": "0" * 64,
            }], checkpoint_for(spec))

        changed_panel = problem.pos[0].copy()
        changed_panel[0, 0] = 1 - changed_panel[0, 0]
        changed_problem = Problem(
            problem.problem_id, problem.category, problem.concept,
            (changed_panel,) + problem.pos[1:], problem.neg)
        wrong_panel_spec = SR.build_runspec(
            opaque_id="problem_00",
            problem=changed_problem,
            cones=[hypothesis],
            registry=default_registry(),
            verifier=verify_hypothesis,
            policy=SR.VerifierPolicy(unexecuted_checks=(
                "contrast", "counterfactual", "archive_regression")),
            expected_verifications={
                hypothesis.hypothesis_id: verification.to_dict()},
            provenance=protocol_provenance(selection_evidence),
            verifier_sources=verifier_related_sources(),
        )
        SR.save_runspec(
            os.path.join(out, "replay_specs", "problem_00.json"),
            wrong_panel_spec,
            allowed_root=td,
        )
        with pytest.raises(SA.ReplayCertificationError,
                           match="corpus/panel identity"):
            SA._cold_replay_specs(out, [{
                **promoted[0],
                "runspec_digest": wrong_panel_spec.spec_digest,
            }], checkpoint_for(wrong_panel_spec))
        SR.save_runspec(
            os.path.join(out, "replay_specs", "problem_00.json"),
            spec,
            allowed_root=td,
        )

        shuffled = PD.build_shuffled_sides_control(
            [manifest_problem], corpus_manifest, seed=23)
        control_manifest = shuffled.manifest
        control_entry = control_manifest["problems"][0]
        controlled_problem = shuffled.problems[0]
        controlled_verification = verify_hypothesis(
            hypothesis, default_registry(), controlled_problem)
        controlled_selection_evidence = _selection_evidence(
            [controlled_verification], [hypothesis.to_dict()],
            controlled_verification, 0.02)
        controlled_selection = _selection_record(
            controlled_verification, [controlled_verification], 0.02)
        tampered_control_entry = {
            **control_entry,
            "assignment": list(reversed(control_entry["assignment"])),
        }
        shuffled_provenance = {
            "runner": "unit_test",
            "dataset": {
                "source": corpus_manifest["sampling"]["source"],
                "seed": corpus_manifest["sampling"]["seed"],
                "limit_per_source": corpus_manifest["sampling"][
                    "limit_per_source"],
                "count_policy": PD.COUNT_POLICY,
                "order_policy": corpus_manifest["sampling"]["order_policy"],
                "repository_commit": corpus_manifest["sampling"][
                    "dataset_revision"],
                "corpus_digest": corpus_digest,
                "corpus_bundle_digest": corpus_bundle["bundle_digest"],
                "panel_set_digest": control_entry[
                    "controlled_panel_set_digest"],
                "panels": "self-contained; source identifier redacted",
            },
            "experiment": {
                "track": "SEMANTIC-PURE",
                "condition": "shuffled-sides",
                "sharing_policy": "shared",
                "control": {
                    "schema": control_manifest["schema"],
                    "control_digest": control_manifest["control_digest"],
                    "base_corpus_digest": control_manifest[
                        "base_corpus_digest"],
                    "seed": control_manifest["seed"],
                    "replicate": control_manifest["replicate"],
                    "assignment_policy": control_manifest[
                        "assignment_policy"],
                    "problem_assignment": tampered_control_entry,
                },
            },
            "proposer": {
                "kind": "offline-test",
                "model": "offline",
                "round_limit": 1,
            },
            "selection": controlled_selection_evidence,
        }
        wrong_assignment_spec = SR.build_runspec(
            opaque_id="problem_00",
            problem=controlled_problem,
            cones=[hypothesis],
            registry=default_registry(),
            verifier=verify_hypothesis,
            policy=SR.VerifierPolicy(unexecuted_checks=(
                "contrast", "counterfactual", "archive_regression")),
            expected_verifications={
                hypothesis.hypothesis_id: controlled_verification.to_dict()},
            provenance=shuffled_provenance,
            verifier_sources=verifier_related_sources(),
        )
        SR.save_runspec(
            os.path.join(out, "replay_specs", "problem_00.json"),
            wrong_assignment_spec,
            allowed_root=td,
        )
        shuffled_checkpoint = {
            "condition": "shuffled-sides",
            "proposer": "offline-test",
            "model": "offline",
            "rounds": 1,
            "control": {
                "control_digest": control_manifest["control_digest"],
            },
            "dataset": {
                "corpus_digest": corpus_digest,
                "corpus_bundle_digest": corpus_bundle["bundle_digest"],
            },
            "records": [{
                "opaque_id": "problem_00",
                "solved": True,
                "track": "SEMANTIC-PURE",
                "condition": "shuffled-sides",
                "sharing_policy": "shared",
                "corpus_digest": corpus_digest,
                "panel_set_digest": control_entry[
                    "controlled_panel_set_digest"],
                "control_digest": control_manifest["control_digest"],
                "replay_spec_digest": wrong_assignment_spec.spec_digest,
            }],
        }
        with pytest.raises(SA.ReplayCertificationError,
                           match="control identity"):
            SA._cold_replay_specs(
                out,
                [{
                    "opaque_id": "problem_00",
                    "hypothesis": hypothesis.to_dict(),
                    "verification": controlled_verification.to_dict(),
                    "selection": controlled_selection,
                    "runspec_digest": wrong_assignment_spec.spec_digest,
                }],
                shuffled_checkpoint,
                corpus_manifest=corpus_manifest,
                corpus_bundle=corpus_bundle,
                control_manifest=control_manifest,
            )
        SR.save_runspec(
            os.path.join(out, "replay_specs", "problem_00.json"),
            spec,
            allowed_root=td,
        )

        with pytest.raises(SA.ReplayCertificationError,
                           match="checkpoint candidates"):
            SA._cold_replay_specs(
                out, promoted,
                checkpoint_for(
                    spec, recorded_candidates=[verification.to_dict()]))

        forged_selection = {
            **selection_record, "conditional_free_energy": -999.0}
        with pytest.raises(SA.ReplayCertificationError,
                           match="promoted selection"):
            SA._cold_replay_specs(out, [{
                **promoted[0], "selection": forged_selection,
            }], checkpoint_for(spec, recorded_selection=forged_selection))

        tolerant_spec = SR.build_runspec(
            opaque_id="problem_00",
            problem=problem,
            cones=[hypothesis],
            registry=default_registry(),
            verifier=verify_hypothesis,
            policy=SR.VerifierPolicy(
                max_support_errors=1,
                unexecuted_checks=(
                    "contrast", "counterfactual", "archive_regression")),
            expected_verifications={
                hypothesis.hypothesis_id: verification.to_dict()},
            provenance=protocol_provenance(selection_evidence),
            verifier_sources=verifier_related_sources(),
        )
        SR.save_runspec(
            os.path.join(out, "replay_specs", "problem_00.json"),
            tolerant_spec,
            allowed_root=td,
        )
        with pytest.raises(SA.ReplayCertificationError, match="exact verifier"):
            SA._cold_replay_specs(out, [{
                **promoted[0], "runspec_digest": tolerant_spec.spec_digest,
            }], checkpoint_for(tolerant_spec))

        # Selection replay must verify every candidate, including a losing
        # candidate that is not the promoted cone.
        tampered_records = [
            dict(record) for record in selection_evidence["candidates"]]
        tampered_records[1]["expected_verification"] = {
            **tampered_records[1]["expected_verification"],
            "support_errors": 1,
        }
        tampered_selection = {
            **selection_evidence,
            "candidates": tampered_records,
        }
        tampered_spec = SR.build_runspec(
            opaque_id="problem_00",
            problem=problem,
            cones=[hypothesis],
            registry=default_registry(),
            verifier=verify_hypothesis,
            policy=SR.VerifierPolicy(unexecuted_checks=(
                "contrast", "counterfactual", "archive_regression")),
            expected_verifications={
                hypothesis.hypothesis_id: verification.to_dict()},
            provenance=protocol_provenance(tampered_selection),
            verifier_sources=verifier_related_sources(),
        )
        SR.save_runspec(
            os.path.join(out, "replay_specs", "problem_00.json"),
            tampered_spec,
            allowed_root=td,
        )
        with pytest.raises(
                SA.ReplayCertificationError,
                match="terminal evidence|cold replay failed"):
            SA._cold_replay_specs(out, [{
                **promoted[0], "runspec_digest": tampered_spec.spec_digest,
            }], checkpoint_for(
                tampered_spec,
                recorded_candidates=[
                    candidate["expected_verification"]
                    for candidate in tampered_records],
            ))
        SR.save_runspec(
            os.path.join(out, "replay_specs", "problem_00.json"),
            spec,
            allowed_root=td,
        )

        old_lab = SA.LAB_DIR
        SA.LAB_DIR = td
        try:
            dest = SA.snapshot_wip("unittest", out, "problem_00")
            assert os.path.exists(
                os.path.join(dest, "problem_00_round00_proposal.txt"))
            promotion_checkpoint = checkpoint_for(spec)
            promotion_record = ProblemResult(
                **promotion_checkpoint["records"][0])
            harness_problem = Problem(
                problem.problem_id, "basic", "harness-only",
                problem.pos, problem.neg)
            art = SA.promote(
                "unittest", out, promotion_checkpoint,
                {"problem_00": _result_payload(
                    harness_problem, promotion_record)},
                promoted)
            assert os.path.exists(os.path.join(art, "results.json"))
            assert os.path.exists(os.path.join(art, "README.md"))
            assert os.path.exists(os.path.join(art, "promoted_cones.json"))
            assert os.path.exists(
                os.path.join(art, "replay_specs", "problem_00.json"))
            assert os.path.exists(
                os.path.join(art, "replay_receipts", "problem_00.json"))
        finally:
            SA.LAB_DIR = old_lab


def test_kolmogorov_selection_keeps_risk_and_complexity_separate():
    simple = CandidateEvaluation(
        "simple",
        Track.SEMANTIC_PURE,
        True,
        RiskVector(R_support=0.2, R_rotated_LOO=0.2),
        ComplexityBreakdown(diagram_node_cost=2),
    )
    better = CandidateEvaluation(
        "better",
        Track.SEMANTIC_PURE,
        True,
        RiskVector(R_support=0.0, R_rotated_LOO=0.0),
        ComplexityBreakdown(diagram_node_cost=3),
    )
    frontier = pareto_frontier([simple, better])
    assert {c.candidate_id for c in frontier} == {"simple", "better"}


# Adversarial score-operator regressions.  These fixtures deliberately keep
# the visual layer elementary: each test is about the meaning of the declared
# score constraint, not about whether a difficult primitive can be extracted.


def _operator_component_panel(component_count: int, offset: int) -> np.ndarray:
    panel = np.zeros((128, 128), dtype=np.uint8)
    locations = ((20, 20), (20, 65), (65, 20), (65, 65))
    for y, x in locations[:component_count]:
        panel[y + offset:y + offset + 12, x:x + 12] = 1
    return panel


def _operator_count_problem(
        positive_counts: tuple[int, ...],
        negative_counts: tuple[int, ...]) -> Problem:
    assert len(positive_counts) == len(negative_counts) == 6
    offsets = (-3, -2, -1, 0, 1, 2)
    return Problem(
        "operator_count", "fixture", "score_operator",
        tuple(_operator_component_panel(count, offset)
              for count, offset in zip(positive_counts, offsets)),
        tuple(_operator_component_panel(count, offset)
              for count, offset in zip(negative_counts, offsets)),
    )


def _operator_count_hypothesis(
        requirement: str, order: str,
        *, description_requirement: str | None = None) -> SemanticHypothesis:
    description_requirement = description_requirement or requirement
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id=(
            "operator_" + re.sub(
                r"[^a-z0-9]+", "_", requirement.lower()).strip("_")),
        description=f"Positive panels have {description_requirement}.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("score", LegCall("object_count", ("scene",))),
        )),
        score_node="score",
        order=order,
        preservation_morphisms=_morphism(),
        semantic_requirements=(requirement,),
    )


def _operator_aspect_problem(
        positive_width: int, negative_width: int) -> Problem:
    def rectangle(width: int, offset: int) -> np.ndarray:
        panel = np.zeros((128, 128), dtype=np.uint8)
        panel[50 + offset:70 + offset, 30:30 + width] = 1
        return panel

    return Problem(
        "operator_aspect", "fixture", "score_operator",
        tuple(rectangle(positive_width, offset)
              for offset in range(-3, 3)),
        tuple(rectangle(negative_width, offset)
              for offset in range(-3, 3)),
    )


def _operator_bbox_hypothesis(
        requirement: str, score_leg: str = "bbox_aspect",
        order: str = "high_positive") -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id=f"operator_{score_leg}_{len(requirement)}",
        description=f"Positive objects have {requirement}.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge("score", LegCall(score_leg, ("main",))),
        )),
        score_node="score",
        order=order,
        preservation_morphisms=_morphism(),
        semantic_requirements=(requirement,),
    )


def _operator_open_closed_problem(
        *, positive_closed: bool) -> Problem:
    def panel(closed: bool, offset: int) -> np.ndarray:
        if closed:
            return _panel_from_polylines([[
                (32 + offset, 32), (92 + offset, 32),
                (92 + offset, 92), (32 + offset, 92),
                (32 + offset, 32),
            ]])
        return _panel_from_polylines([[
            (25 + offset, 64), (103 + offset, 64),
        ]])

    return Problem(
        "operator_closedness", "fixture", "score_operator",
        tuple(panel(positive_closed, offset) for offset in range(-3, 3)),
        tuple(panel(not positive_closed, offset) for offset in range(-3, 3)),
    )


def _operator_closedness_hypothesis(
        requirement: str, order: str) -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id=(
            "operator_" + re.sub(
                r"[^a-z0-9]+", "_", requirement.lower()).strip("_")),
        description=f"Positive objects are {requirement}.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge(
                "contour", LegCall("extract_contours", ("main",))),
            DiagramEdge(
                "score", LegCall("contour_closedness", ("contour",))),
        )),
        score_node="score",
        order=order,
        preservation_morphisms=_morphism(),
        semantic_requirements=(requirement,),
    )


def _operator_triangle_problem(
        *, positive_triangles: bool) -> Problem:
    def triangle(offset: int) -> np.ndarray:
        return _panel_from_polylines([[
            (30 + offset, 32), (94 + offset, 32),
            (62 + offset, 88), (30 + offset, 32),
        ]])

    def square(offset: int) -> np.ndarray:
        return _panel_from_polylines([[
            (32 + offset, 32), (92 + offset, 32),
            (92 + offset, 92), (32 + offset, 92),
            (32 + offset, 32),
        ]])

    return Problem(
        "operator_triangle", "fixture", "score_operator",
        tuple((triangle if positive_triangles else square)(offset)
              for offset in range(-3, 3)),
        tuple((square if positive_triangles else triangle)(offset)
              for offset in range(-3, 3)),
    )


def _operator_triangle_hypothesis(requirement: str) -> SemanticHypothesis:
    return replace(
        _triangle_witness_hypothesis(),
        hypothesis_id=(
            "operator_" + re.sub(
                r"[^a-z0-9]+", "_", requirement.lower()).strip("_")),
        description=f"Positive panels have {requirement}.",
        semantic_requirements=(requirement,),
    )


def _operator_line_problem(
        positive_paths: tuple[tuple[tuple[int, int], ...], ...],
        negative_paths: tuple[tuple[tuple[int, int], ...], ...]) -> Problem:
    assert len(positive_paths) == len(negative_paths) == 6
    return Problem(
        "operator_line", "fixture", "score_operator",
        tuple(_panel_from_polylines([path]) for path in positive_paths),
        tuple(_panel_from_polylines([path]) for path in negative_paths),
    )


def _operator_line_hypothesis(
        requirement: str, score_leg: str, order: str) -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id=f"operator_{score_leg}_{len(requirement)}",
        description=f"Positive objects have {requirement}.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge(
                "contour", LegCall("extract_contours", ("main",))),
            DiagramEdge(
                "line", LegCall("fit_line_segment", ("contour",))),
            DiagramEdge("score", LegCall(score_leg, ("line",))),
        )),
        score_node="score",
        order=order,
        preservation_morphisms=_morphism(),
        semantic_requirements=(requirement,),
    )


def _operator_failure_text(result) -> str:
    return f"{result.compile_error} {result.semantic_issue}".lower()


def test_header_calibration_binds_each_number_to_its_own_claim():
    hypothesis = replace(
        _object_count_hypothesis(),
        hypothesis_id="swapped_part_contact_cardinals",
        description="Positive panels have two parts and three contacts.",
        semantic_requirements=("three parts", "two contacts"),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert "calibration" in result.compile_error.lower()


def test_header_calibration_rejects_mode_mismatch_for_same_cardinal():
    hypothesis = _operator_count_hypothesis(
        "two components", "high_positive",
        description_requirement="at least two components")
    result = verify_hypothesis(
        hypothesis, default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6))
    assert not result.accepted
    assert "calibration" in result.compile_error.lower()


@pytest.mark.parametrize(
    ("description_requirement", "requirement", "order",
     "positive_counts", "negative_counts"),
    (
        ("at least two components", "two or more components",
         "high_positive", (2, 3, 2, 3, 2, 3), (1,) * 6),
        ("at most two components", "two or fewer components",
         "low_positive", (1, 2, 1, 2, 1, 2), (3,) * 6),
    ),
)
def test_header_calibration_accepts_normalized_equivalent_modes(
        description_requirement, requirement, order,
        positive_counts, negative_counts):
    result = verify_hypothesis(
        _operator_count_hypothesis(
            requirement, order,
            description_requirement=description_requirement),
        default_registry(),
        _operator_count_problem(positive_counts, negative_counts),
    )
    assert result.accepted, _operator_failure_text(result)


@pytest.mark.parametrize(
    ("description_requirement", "requirement"),
    (("exactly two objects", "exactly two components"),
     ("exactly two components", "exactly two objects")),
)
def test_header_accepts_score_contract_aliases(
        description_requirement, requirement):
    result = verify_hypothesis(
        _operator_count_hypothesis(
            requirement, "high_positive",
            description_requirement=description_requirement),
        default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert result.accepted, _operator_failure_text(result)


@pytest.mark.parametrize(
    ("description_requirement", "requirement"),
    (("no components", "zero components"),
     ("zero components", "without components")),
)
def test_header_accepts_equivalent_zero_count_forms(
        description_requirement, requirement):
    result = verify_hypothesis(
        _operator_count_hypothesis(
            requirement, "low_positive",
            description_requirement=description_requirement),
        default_registry(),
        _operator_count_problem((0,) * 6, (1,) * 6),
    )
    assert result.accepted, _operator_failure_text(result)


@pytest.mark.parametrize(
    ("description_requirement", "requirement"),
    (("an open contour", "not closed contour"),
     ("an unclosed contour", "open contour")),
)
def test_header_accepts_equivalent_binary_proxy_polarities(
        description_requirement, requirement):
    hypothesis = replace(
        _operator_closedness_hypothesis(requirement, "low_positive"),
        description=f"Positive objects have {description_requirement}.",
    )
    result = verify_hypothesis(
        hypothesis, default_registry(),
        _operator_open_closed_problem(positive_closed=False),
    )
    assert result.accepted, _operator_failure_text(result)


def test_binary_not_closed_inverts_both_target_and_score_direction():
    valid = verify_hypothesis(
        _operator_closedness_hypothesis("not closed", "low_positive"),
        default_registry(),
        _operator_open_closed_problem(positive_closed=False),
    )
    assert valid.accepted, _operator_failure_text(valid)

    inverted = verify_hypothesis(
        _operator_closedness_hypothesis("not closed", "high_positive"),
        default_registry(),
        _operator_open_closed_problem(positive_closed=True),
    )
    assert not inverted.accepted
    assert inverted.semantic_issue or inverted.compile_error


def test_binary_measurement_cannot_claim_two_loops():
    result = verify_hypothesis(
        _operator_closedness_hypothesis("two loops", "high_positive"),
        default_registry(),
        _operator_open_closed_problem(positive_closed=True),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


def test_continuous_numeric_comparator_is_an_absolute_constraint():
    invalid = verify_hypothesis(
        _operator_bbox_hypothesis("an aspect ratio above two"),
        default_registry(), _operator_aspect_problem(36, 24))
    assert not invalid.accepted
    assert invalid.semantic_issue or invalid.compile_error

    valid = verify_hypothesis(
        _operator_bbox_hypothesis("an aspect ratio above two"),
        default_registry(), _operator_aspect_problem(52, 36))
    assert valid.accepted, _operator_failure_text(valid)


def test_structural_cardinal_requires_a_count_of_that_witness():
    result = verify_hypothesis(
        _operator_triangle_hypothesis("two triangles"),
        default_registry(),
        _operator_triangle_problem(positive_triangles=True),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


@pytest.mark.parametrize("requirement", ("object", "higher measurement"))
def test_continuous_score_requires_an_associated_metric_claim(requirement):
    result = verify_hypothesis(
        _operator_bbox_hypothesis(requirement),
        default_registry(), _operator_aspect_problem(48, 20))
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


@pytest.mark.parametrize(
    ("requirement", "positive_counts", "negative_counts"),
    (
        ("a pair of components", (1,) * 6, (0,) * 6),
        ("multiple components", (1,) * 6, (0,) * 6),
        ("many components", (1,) * 6, (0,) * 6),
        ("few components", (1,) * 6, (0,) * 6),
        ("several components", (1,) * 6, (0,) * 6),
    ),
)
def test_count_quantifiers_do_not_collapse_to_presence(
        requirement, positive_counts, negative_counts):
    result = verify_hypothesis(
        _operator_count_hypothesis(requirement, "high_positive"),
        default_registry(),
        _operator_count_problem(positive_counts, negative_counts),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


@pytest.mark.parametrize(
    ("requirement", "positive_counts", "negative_counts"),
    (
        ("a pair of components", (2,) * 6, (1,) * 6),
        ("multiple components", (2, 3, 2, 3, 2, 3), (1,) * 6),
    ),
)
def test_supported_count_quantifiers_have_fixed_meanings(
        requirement, positive_counts, negative_counts):
    result = verify_hypothesis(
        _operator_count_hypothesis(requirement, "high_positive"),
        default_registry(),
        _operator_count_problem(positive_counts, negative_counts),
    )
    assert result.accepted, _operator_failure_text(result)


@pytest.mark.parametrize(
    ("requirement", "order", "positive_counts", "negative_counts"),
    (
        ("not more than two components", "low_positive",
         (1, 2, 1, 2, 1, 2), (3,) * 6),
        ("no less than two components", "high_positive",
         (2, 3, 2, 3, 2, 3), (1,) * 6),
        ("two or more components", "high_positive",
         (2, 3, 2, 3, 2, 3), (1,) * 6),
        ("two or fewer components", "low_positive",
         (1, 2, 1, 2, 1, 2), (3,) * 6),
    ),
)
def test_count_comparators_are_canonicalized_before_negation(
        requirement, order, positive_counts, negative_counts):
    result = verify_hypothesis(
        _operator_count_hypothesis(requirement, order),
        default_registry(),
        _operator_count_problem(positive_counts, negative_counts),
    )
    assert result.accepted, _operator_failure_text(result)


@pytest.mark.parametrize(
    ("requirement", "order", "positive_counts", "negative_counts"),
    (
        (">= 2 components", "high_positive", (2,) * 6, (1,) * 6),
        ("<= 2 components", "low_positive", (2,) * 6, (3,) * 6),
        ("> 2 components", "high_positive", (2,) * 6, (1,) * 6),
        ("< 2 components", "low_positive", (2,) * 6, (3,) * 6),
        ("!= 2 components", "high_positive", (2,) * 6, (1,) * 6),
        ("≤ 2 components", "low_positive", (2,) * 6, (3,) * 6),
        ("+2 components", "high_positive", (2,) * 6, (1,) * 6),
        ("-2 components", "high_positive", (2,) * 6, (1,) * 6),
        ("1e3 components", "high_positive", (3,) * 6, (2,) * 6),
        ("1.0 components", "high_positive", (1,) * 6, (0,) * 6),
        (("9" * 400) + " components", "high_positive",
         (2,) * 6, (1,) * 6),
    ),
    ids=("ascii-gte", "ascii-lte", "ascii-gt", "ascii-lt",
         "ascii-ne", "unicode-lte", "explicit-plus",
         "negative-cardinal", "scientific-notation", "decimal",
         "huge-cardinal"),
)
def test_unsupported_numeric_syntax_fails_closed(
        requirement, order, positive_counts, negative_counts):
    result = verify_hypothesis(
        _operator_count_hypothesis(requirement, order),
        default_registry(),
        _operator_count_problem(positive_counts, negative_counts),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


def test_exact_two_uses_a_non_monotone_fixed_predicate():
    result = verify_hypothesis(
        _operator_count_hypothesis("two components", "high_positive"),
        default_registry(),
        _operator_count_problem(
            (2,) * 6, (1, 3, 1, 3, 1, 3)),
    )
    assert result.accepted, _operator_failure_text(result)
    assert result.support_errors == 0
    assert result.loo_errors == 0
    assert result.rotated_loo_errors == 0


def test_no_triangle_inverts_witness_presence():
    hypothesis = _operator_triangle_hypothesis("no triangle")
    inverted = verify_hypothesis(
        hypothesis, default_registry(),
        _operator_triangle_problem(positive_triangles=True),
    )
    assert not inverted.accepted
    assert inverted.semantic_issue or inverted.compile_error


@pytest.mark.parametrize("requirement", ("a straight line", "a straighter line"))
def test_straight_proxy_direction_is_a_calibrated_metric_claim(requirement):
    straight = tuple(
        ((20 + offset, 64), (108 + offset, 64))
        for offset in range(-3, 3))
    gently_bent = tuple(
        ((20 + offset, 64), (45 + offset, 62),
         (70 + offset, 64), (108 + offset, 66))
        for offset in range(-3, 3))
    result = verify_hypothesis(
        _operator_line_hypothesis(
            requirement, "line_residual", "low_positive"),
        default_registry(),
        _operator_line_problem(straight, gently_bent),
    )
    assert result.accepted, _operator_failure_text(result)


def _operator_occupancy_problem() -> Problem:
    def filled(offset: int) -> np.ndarray:
        panel = np.zeros((128, 128), dtype=np.uint8)
        panel[42 + offset:82 + offset, 30:94] = 1
        return panel

    def outline(offset: int) -> np.ndarray:
        return _panel_from_polylines([[
            (30 + offset, 42), (94 + offset, 42),
            (94 + offset, 82), (30 + offset, 82),
            (30 + offset, 42),
        ]])

    return Problem(
        "operator_occupancy", "fixture", "score_operator",
        tuple(filled(offset) for offset in range(-3, 3)),
        tuple(outline(offset) for offset in range(-3, 3)),
    )


@pytest.mark.parametrize(
    "case",
    ("length_for_residual", "residual_for_length",
     "aspect_for_occupancy", "occupancy_for_aspect"),
)
def test_metric_words_cannot_be_laundered_through_a_different_score(case):
    long_lines = tuple(
        ((18 + offset, 60), (110 + offset, 60))
        for offset in range(-3, 3))
    short_lines = tuple(
        ((36 + offset, 60), (88 + offset, 60))
        for offset in range(-3, 3))
    straight = tuple(
        ((20 + offset, 64), (108 + offset, 64))
        for offset in range(-3, 3))
    gently_bent = tuple(
        ((20 + offset, 64), (45 + offset, 62),
         (70 + offset, 64), (108 + offset, 66))
        for offset in range(-3, 3))

    if case == "length_for_residual":
        hypothesis = _operator_line_hypothesis(
            "higher line residual and length",
            "line_length", "high_positive")
        problem = _operator_line_problem(long_lines, short_lines)
    elif case == "residual_for_length":
        hypothesis = _operator_line_hypothesis(
            "higher line length and residual",
            "line_residual", "high_positive")
        problem = _operator_line_problem(gently_bent, straight)
    elif case == "aspect_for_occupancy":
        hypothesis = _operator_bbox_hypothesis(
            "a higher aspect ratio and occupancy", "bbox_aspect")
        problem = _operator_aspect_problem(48, 20)
    else:
        hypothesis = _operator_bbox_hypothesis(
            "higher occupancy and aspect ratio", "bbox_occupancy")
        problem = _operator_occupancy_problem()

    result = verify_hypothesis(hypothesis, default_registry(), problem)
    assert not result.accepted, hypothesis.hypothesis_id
    assert result.semantic_issue or result.compile_error


def test_header_cannot_normalize_length_into_residual():
    hypothesis = replace(
        _operator_line_hypothesis(
            "higher line residual", "line_residual", "high_positive"),
        description="Positive figures have longer lines.",
    )
    straight = tuple(
        ((20 + offset, 64), (108 + offset, 64))
        for offset in range(-3, 3))
    gently_bent = tuple(
        ((20 + offset, 64), (45 + offset, 62),
         (70 + offset, 64), (108 + offset, 66))
        for offset in range(-3, 3))
    result = verify_hypothesis(
        hypothesis, default_registry(),
        _operator_line_problem(gently_bent, straight),
    )
    assert not result.accepted
    assert "calibration" in result.compile_error.lower()


@pytest.mark.parametrize("requirement", ("nonclosed contour", "unclosed curve"))
def test_prefixed_binary_negation_is_not_erased(requirement):
    result = verify_hypothesis(
        _operator_closedness_hypothesis(requirement, "low_positive"),
        default_registry(),
        _operator_open_closed_problem(positive_closed=False),
    )
    assert result.accepted, _operator_failure_text(result)


def test_quantified_object_is_retained_for_object_count():
    result = verify_hypothesis(
        _operator_count_hypothesis("two objects", "high_positive"),
        default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert result.accepted, _operator_failure_text(result)


@pytest.mark.parametrize("requirement", (
    "roughly two components",
    "maximum two components",
    "minimum two components",
    "larger components",
    "part above part",
))
def test_unconsumed_quantity_or_spatial_language_fails_closed(requirement):
    result = verify_hypothesis(
        _operator_count_hypothesis(requirement, "high_positive"),
        default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


def test_fixed_count_constraints_use_conjunction_semantics():
    hypothesis = replace(
        _operator_count_hypothesis(
            "at least two components", "high_positive"),
        description=(
            "Positive panels have at least two components and not three "
            "components."),
        semantic_requirements=(
            "at least two components", "not three components"),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert result.accepted, _operator_failure_text(result)
    assert result.rule == "score>=2 & score!=3"


def test_direct_witness_presence_requires_the_final_witness_claim():
    hypothesis = replace(
        _triangle_witness_hypothesis(),
        hypothesis_id="polygon_is_not_triangle_presence",
        description="The principal object is polygonal.",
        semantic_requirements=("polygon",),
        witness_requirements=("PolygonWitness",),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(),
        _operator_triangle_problem(positive_triangles=True),
    )
    assert not result.accepted
    assert result.semantic_issue \
        == "semantic_measurement_has_no_calibrated_claim"


def test_direct_witness_absence_is_an_executable_fixed_predicate():
    result = verify_hypothesis(
        _operator_triangle_hypothesis("no triangle"),
        default_registry(),
        _operator_triangle_problem(positive_triangles=False),
    )
    assert result.accepted, _operator_failure_text(result)
    assert result.rule == "score:absent"
    assert result.loo_errors == result.rotated_loo_errors == 0


@pytest.mark.parametrize(
    ("requirement", "order", "positive_counts", "negative_counts"),
    (
        ("not greater than two components", "low_positive",
         (1, 2, 1, 2, 1, 2), (3,) * 6),
        ("no fewer than two components", "high_positive",
         (2, 3, 2, 3, 2, 3), (1,) * 6),
        ("not at least two components", "low_positive",
         (1,) * 6, (2,) * 6),
        ("not at most two components", "high_positive",
         (3,) * 6, (2,) * 6),
    ),
)
def test_negated_bounds_are_canonicalized_logically(
        requirement, order, positive_counts, negative_counts):
    result = verify_hypothesis(
        _operator_count_hypothesis(requirement, order),
        default_registry(),
        _operator_count_problem(positive_counts, negative_counts),
    )
    assert result.accepted, _operator_failure_text(result)


@pytest.mark.parametrize("requirement", (
    "2x components", "2nd component", "~2 components", "≈2 components",
    "2? components",
))
def test_alphanumeric_numeric_junk_cannot_degrade_to_a_cardinal(requirement):
    result = verify_hypothesis(
        _operator_count_hypothesis(requirement, "high_positive"),
        default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


def test_count_claim_identity_includes_its_counted_subject():
    registry = default_registry()
    contact_count = registry.get("contact_count")
    assert term_matches_contract_claim("contact count is two", contact_count)
    assert not term_matches_contract_claim("part count is two", contact_count)
    assert calibrated_claim_signature("part count is two") == (
        (("part", "count"), "exact", 2, None, False),)
    assert calibrated_claim_signature("two parts in contact") == (
        (("part",), "exact", 2, None, False),)
    assert calibrated_claim_signature("two circle intersections") == (
        (("intersection",), "exact", 2, None, False),)
    assert calibrated_claim_signature(
        "number of parts contacting others is two") == (
            (("part", "count"), "exact", 2, None, False),)
    object_count = registry.get("object_count")
    assert not term_matches_contract_claim("country", object_count)
    assert not term_matches_contract_claim("counter", object_count)
    assert term_matches_contract_claim(
        "three branches", registry.get("branch_count"))


def test_cardinal_cannot_slide_from_subject_to_later_relation():
    hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id="two_parts_is_not_two_contacts",
        description="Positive panels have two parts in contact.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("graph", LegCall("build_part_graph", ("scene",))),
            DiagramEdge(
                "score", LegCall("contact_count", ("graph",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("two parts in contact",),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert result.semantic_issue == "MISSING_LEG" \
        or "not bound to the final measurement" in result.compile_error


def test_two_sided_fixed_interval_does_not_create_direction_conflict():
    hypothesis = replace(
        _operator_count_hypothesis(
            "at least two components", "high_positive"),
        description=(
            "Positive panels have at least two components and at most four "
            "components."),
        semantic_requirements=(
            "at least two components", "at most four components"),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(),
        _operator_count_problem((2, 3, 4, 2, 3, 4), (1,) * 6),
    )
    assert result.accepted, _operator_failure_text(result)
    assert result.rule == "score>=2 & score<=4"


def test_shared_head_fixed_interval_binds_both_clauses():
    requirement = "at least two and at most four components"
    assert calibrated_claim_signature(requirement) == (
        (("component",), "at_least", 2, "high", False),
        (("component",), "at_most", 4, "low", False),
    )
    result = verify_hypothesis(
        _operator_count_hypothesis(requirement, "high_positive"),
        default_registry(),
        _operator_count_problem((2, 3, 4, 2, 3, 4), (1,) * 6),
    )
    assert result.accepted, _operator_failure_text(result)
    assert result.rule == "score>=2 & score<=4"


def test_redundant_binary_proxy_conjunction_resolves_clause_locally():
    result = verify_hypothesis(
        _operator_closedness_hypothesis(
            "not closed but open contour", "low_positive"),
        default_registry(),
        _operator_open_closed_problem(positive_closed=False),
    )
    assert result.accepted, _operator_failure_text(result)
    assert result.rule == "score==0"


def _operator_polygon_side_hypothesis(
        requirement: str, order: str = "low_positive") -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id=(
            "operator_polygon_" + re.sub(
                r"[^a-z0-9]+", "_", requirement.lower()).strip("_")),
        description=f"Positive panels have {requirement}.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge(
                "contour", LegCall("extract_contours", ("main",))),
            DiagramEdge(
                "polygon", LegCall("fit_polygon", ("contour",))),
            DiagramEdge(
                "score", LegCall("polygon_side_count", ("polygon",))),
        )),
        score_node="score",
        order=order,
        preservation_morphisms=_morphism(),
        semantic_requirements=(requirement,),
    )


def test_count_contracts_match_the_counted_head_not_the_carrier():
    registry = default_registry()
    polygon_sides = registry.get("polygon_side_count")
    assert term_matches_contract_claim("three sides", polygon_sides)
    assert not term_matches_contract_claim("three polygons", polygon_sides)
    assert not term_matches_contract_claim(
        "two paths", registry.get("endpoint_count"))
    assert not term_matches_contract_claim(
        "two curves", registry.get("count_inflections"))


def test_polygon_side_count_cannot_turn_three_polygons_into_three_sides():
    result = verify_hypothesis(
        _operator_polygon_side_hypothesis("three polygons"),
        default_registry(),
        _operator_triangle_problem(positive_triangles=True),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


@pytest.mark.parametrize("requirement", (
    "more overall ink", "more aggregate ink",
))
def test_aggregate_ink_aliases_belong_only_to_total_ink(requirement):
    registry = default_registry()
    assert term_matches_contract_claim(requirement, registry.get("total_ink"))
    assert not term_matches_contract_claim(
        requirement, registry.get("largest_ink"))


@pytest.mark.parametrize("surface", ("@", "^", "&", "🙂", "½"))
def test_discarded_object_count_surface_fails_closed(surface):
    requirement = f"two {surface} components"
    result = verify_hypothesis(
        _operator_count_hypothesis(requirement, "high_positive"),
        default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


@pytest.mark.parametrize("modifier", (
    "point-shaped", "simple", "plain", "basic",
))
def test_substantive_modifiers_cannot_vanish_as_framing(modifier):
    requirement = f"two {modifier} components"
    result = verify_hypothesis(
        _operator_count_hypothesis(requirement, "high_positive"),
        default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


@pytest.mark.parametrize("requirement", (
    "non-closed contour", "un-closed curve",
))
def test_hyphenated_prefixed_closedness_preserves_negation(requirement):
    result = verify_hypothesis(
        _operator_closedness_hypothesis(requirement, "low_positive"),
        default_registry(),
        _operator_open_closed_problem(positive_closed=False),
    )
    assert result.accepted, _operator_failure_text(result)


def test_other_than_cardinal_executes_not_exact():
    assert calibrated_claim_signature("other than two components") == (
        (("component",), "not_exact", 2, None, True),)
    result = verify_hypothesis(
        _operator_count_hypothesis(
            "other than two components", "high_positive"),
        default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert not result.accepted
    assert result.semantic_issue == "semantic_count_positive_violates_not_exact"


def test_lexicalized_less_words_distinguish_countless_from_contactless():
    countless = parse_score_operator("countless components")
    contactless = parse_score_operator("contactless")
    assert countless.mode == "unsupported"
    assert contactless.mode == "absence"
    assert contactless.negated

    result = verify_hypothesis(
        _operator_count_hypothesis(
            "countless components", "high_positive"),
        default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


def test_number_of_phrase_binds_to_polygon_side_count():
    result = verify_hypothesis(
        _operator_polygon_side_hypothesis(
            "number of polygon sides is three"),
        default_registry(),
        _operator_triangle_problem(positive_triangles=True),
    )
    assert result.accepted, _operator_failure_text(result)
    assert result.rule == "score==3"


def test_number_of_phrase_binds_to_connected_component_count():
    result = verify_hypothesis(
        _operator_count_hypothesis(
            "number of connected components is two", "high_positive"),
        default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert result.accepted, _operator_failure_text(result)
    assert result.rule == "score==2"


def test_decorative_scene_parse_does_not_semanticize_total_ink():
    hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id="decorative_scene_total_ink",
        description="Positive panels have more overall ink.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("score", LegCall("total_ink", ("panel",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("more overall ink",),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(), _problem_two_objects_vs_one())
    assert not result.accepted
    assert result.semantic_issue == "measurement_only_direct_panel_statistic"


def test_cofibration_handles_populated_ndarray_fields():
    source = {"mask": np.array([[1, 0], [0, 1]], dtype=np.uint8)}
    target = {
        "carrier": {"mask": source["mask"].copy()},
        "interface": np.array([1], dtype=np.uint8),
        "patch": np.array([2], dtype=np.uint8),
    }
    spec = CofibrationSpec(
        name="ndarray_gluing",
        source_type="dict",
        target_type="dict",
        interface_fields=("interface",),
        added_fields=("patch",),
        attachment_leg="fixture_attachment",
    )
    check = verify_cofibration(source, target, spec)
    # Array truth-value ambiguity must not escape; this deliberately generic
    # fixture has no endpoint-bearing relation and therefore fails binding.
    assert not check.ok
    assert check.first_failed == "attachment_unbound"


def test_cofibration_attachment_must_involve_the_glued_source():
    body = _part_fixture(
        "body", "fixture",
        ((5.0, 5.0), (10.0, 5.0), (15.0, 5.0), (20.0, 5.0)))
    left = _part_fixture("a", "fixture", ((30.0, 20.0), (35.0, 20.0)))
    right = _part_fixture("b", "fixture", ((40.0, 20.0), (45.0, 20.0)))
    unrelated = ContactWitness(
        source_a="a", source_b="b",
        points=(PointWitness(x=37.5, y=20.0),), relation="attachment")
    target = PartGraphWitness(
        parts=(body, left, right), contacts=(unrelated,),
        adjacency=(("a", "b"),))

    check = verify_cofibration(body, target, _gluing_spec())
    assert not check.ok
    assert check.first_failed == "attachment_unbound"


def test_verifier_rejects_attachment_elsewhere_in_the_target_graph():
    def fixture_graph(panel: np.ndarray) -> PartGraphWitness:
        body = _part_fixture(
            "body", "fixture",
            tuple((float(x), 10.0) for x in range(5, 30, 4)))
        left = _part_fixture("a", "fixture", ((40.0, 30.0), (44.0, 30.0)))
        right = _part_fixture("b", "fixture", ((50.0, 30.0), (54.0, 30.0)))
        contacts = ()
        adjacency = ()
        if np.any(panel):
            contacts = (ContactWitness(
                source_a="a", source_b="b",
                points=(PointWitness(x=47.0, y=30.0),),
                relation="attachment"),)
            adjacency = (("a", "b"),)
        return PartGraphWitness(
            parts=(body, left, right), contacts=contacts,
            adjacency=adjacency)

    registry = default_registry()
    registry.register(LegContract(
        name="fixture_isolated_graph",
        domain=("Panel",),
        codomain="PartGraphWitness",
        implementation=fixture_graph,
        complexity=1,
    ))
    hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id="isolated_principal_attachment_elsewhere",
        description="The principal part is attached.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge(
                "graph", LegCall("fixture_isolated_graph", ("panel",))),
            DiagramEdge(
                "principal", LegCall("select_largest_part", ("graph",))),
            DiagramEdge(
                "attachment", LegCall("detect_attachment", ("graph",))),
            DiagramEdge(
                "score", LegCall("contact_confidence", ("attachment",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("principal part attached",),
        cofibrations=(CofibrationSpec(
            name="principal_attached_to_graph",
            source_node="principal",
            target_node="graph",
            source_type="PartWitness",
            target_type="PartGraphWitness",
            interface_fields=("contacts",),
            added_fields=("parts",),
            attachment_leg="detect_attachment",
            projection_leg="select_largest_part",
        ),),
    )
    positive = np.zeros((64, 64), dtype=np.uint8)
    positive[20, 20] = 1
    negative = np.zeros_like(positive)
    problem = Problem(
        "isolated_principal", "fixture", "cofibration_binding",
        tuple(positive.copy() for _ in range(6)),
        tuple(negative.copy() for _ in range(6)),
    )
    result = verify_hypothesis(hypothesis, registry, problem)
    assert not result.accepted
    assert result.support_errors == 0
    assert result.cofibration_errors == 6


def test_partial_translation_applicability_is_unchecked():
    def panel(component_count: int, clipped: bool) -> np.ndarray:
        image = np.zeros((128, 128), dtype=np.uint8)
        image[24:36, 24:36] = 1
        if component_count == 2:
            if clipped:
                image[122:128, 82:94] = 1
            else:
                image[72:84, 82:94] = 1
        elif clipped:
            image[:] = 0
            image[122:128, 82:94] = 1
        return image

    problem = Problem(
        "partial_translation", "fixture", "score_operator",
        tuple(panel(2, index >= 3) for index in range(6)),
        tuple(panel(1, index >= 3) for index in range(6)),
    )
    result = verify_hypothesis(
        _object_count_hypothesis(), default_registry(), problem)
    assert result.declared_morphism_checks > 0
    assert result.unchecked_morphisms
    assert any("translate" in name for name in result.unchecked_morphisms)
    assert result.risk.R_naturality is None
    assert not result.accepted


def test_decomposition_matches_and_suggests_the_decompose_leg():
    registry = default_registry()
    contract = registry.get("decompose_component_into_parts")
    assert term_matches_contract_claim("decomposition", contract)
    assert "decompose_component_into_parts" in leg_suggestions(
        "decomposition", registry)


@pytest.mark.parametrize("requirement", (
    "two components within another component",
    "one component contains another component",
    "two components between other components",
))
def test_unimplemented_relation_prose_fails_closed(requirement):
    result = verify_hypothesis(
        _operator_count_hypothesis(requirement, "high_positive"),
        default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error


def test_binary_open_contour_cardinal_requires_a_count():
    result = verify_hypothesis(
        _operator_closedness_hypothesis("one open contour", "low_positive"),
        default_registry(),
        _operator_open_closed_problem(positive_closed=False),
    )
    assert not result.accepted
    diagnostic = _operator_failure_text(result)
    assert "cardinal" in diagnostic
    assert "requires" in diagnostic and "count" in diagnostic


@pytest.mark.parametrize("requirement", (
    "more open contours", "no open contours", "closed curves",
))
def test_plural_binary_alias_requires_a_count_measurement(requirement):
    result = verify_hypothesis(
        _operator_closedness_hypothesis(requirement, "low_positive"),
        default_registry(),
        _operator_open_closed_problem(positive_closed=False),
    )
    assert not result.accepted
    diagnostic = _operator_failure_text(result)
    assert "plural" in diagnostic and "count" in diagnostic


def test_plural_continuous_proxy_requires_a_count_measurement():
    straight = tuple(
        ((20 + offset, 64), (108 + offset, 64))
        for offset in range(-3, 3))
    bent = tuple(
        ((20 + offset, 64), (45 + offset, 62),
         (70 + offset, 64), (108 + offset, 66))
        for offset in range(-3, 3))
    result = verify_hypothesis(
        _operator_line_hypothesis(
            "more straight lines", "line_residual", "low_positive"),
        default_registry(), _operator_line_problem(straight, bent),
    )
    assert not result.accepted
    diagnostic = _operator_failure_text(result)
    assert "plural" in diagnostic and "count" in diagnostic


@pytest.mark.parametrize("requirement", (
    "not without contact", "not contactless", "not other than two contacts",
    "not unclosed contour",
))
def test_nested_negation_fails_closed(requirement):
    assert parse_score_operator(requirement).mode == "unsupported"


@pytest.mark.parametrize("requirement", (
    "each object has exactly two endpoints",
    "two endpoints per object",
))
def test_distributive_count_claim_cannot_ride_on_largest_object(requirement):
    def positive(offset: int) -> np.ndarray:
        return _panel_from_polylines((
            ((18 + offset, 35), (108 + offset, 35)),
            ((75, 75), (91, 75), (91, 91), (75, 91), (75, 75)),
        ))

    def negative(offset: int) -> np.ndarray:
        return _panel_from_polylines((
            ((18 + offset, 35), (108 + offset, 35)),
            ((63 + offset, 35), (63 + offset, 92)),
        ))

    hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id="distributive_endpoint_scope",
        description=f"Positive panels have {requirement}.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge(
                "skeleton", LegCall("build_skeleton_graph", ("main",))),
            DiagramEdge(
                "score", LegCall("endpoint_count", ("skeleton",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=(requirement,),
    )
    problem = Problem(
        "distributive_scope", "fixture", "score_operator",
        tuple(positive(offset) for offset in range(-3, 3)),
        tuple(negative(offset) for offset in range(-3, 3)),
    )
    result = verify_hypothesis(hypothesis, default_registry(), problem)
    assert not result.accepted
    assert "unsupported" in _operator_failure_text(result)


def test_shared_and_comparator_is_one_claim_but_conjunction_still_splits():
    assert calibrated_claim_signature("two and above components") == (
        (("component",), "at_least", 2, "high", False),)
    shared = verify_hypothesis(
        _operator_count_hypothesis(
            "two and above components", "high_positive"),
        default_registry(),
        _operator_count_problem((2, 3, 2, 3, 2, 3), (1,) * 6),
    )
    assert shared.accepted, _operator_failure_text(shared)

    conjunction = replace(
        _operator_count_hypothesis(
            "at least two components", "high_positive"),
        description=(
            "Positive panels have at least two components and not three "
            "components."),
        semantic_requirements=(
            "at least two components", "not three components"),
    )
    result = verify_hypothesis(
        conjunction, default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert result.accepted, _operator_failure_text(result)
    assert result.rule == "score>=2 & score!=3"


def test_negative_side_description_cannot_invert_positive_satisfies():
    hypothesis = replace(
        _operator_count_hypothesis("two components", "high_positive"),
        description="Negative panels have two components.",
    )
    result = verify_hypothesis(
        hypothesis, default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert not result.accepted
    assert "negative side" in result.compile_error


def test_structured_term_cannot_hide_opposite_side_scope():
    hypothesis = replace(
        _operator_count_hypothesis("two components", "high_positive"),
        semantic_requirements=("Negative panels have two components.",),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert not result.accepted
    assert "side-free predicates" in result.compile_error


def test_negative_comparison_complement_remains_valid_description():
    result = verify_hypothesis(
        _object_count_hypothesis(), default_registry(),
        _problem_two_objects_vs_one(),
    )
    assert result.accepted, _operator_failure_text(result)


@pytest.mark.parametrize("fragment", ("art", "wit", "ness", "gra"))
def test_unknown_fragments_cannot_substring_match_witness_type_names(fragment):
    requirement = f"exactly two {fragment}-like parts"
    hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id=f"substring_laundering_{fragment}",
        description=f"Positive panels have {requirement}.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("graph", LegCall("build_part_graph", ("scene",))),
            DiagramEdge("score", LegCall("part_count", ("graph",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=(requirement,),
    )
    result = verify_hypothesis(
        hypothesis, default_registry(),
        _operator_count_problem((2,) * 6, (1,) * 6),
    )
    assert not result.accepted
    assert result.semantic_issue or result.compile_error
