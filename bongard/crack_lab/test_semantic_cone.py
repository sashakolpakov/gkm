import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import Problem, trace_shape
from semantic_compiler import CompileError, compile_hypothesis
from semantic_ir import DiagramEdge, DiagramSpec, LegCall, MorphSpec, SemanticHypothesis
from semantic_legs import default_registry
from semantic_verifier import verify_hypothesis
from semantic_selection import CandidateEvaluation, ComplexityBreakdown, RiskVector, Track, pareto_frontier
from cofibrations import CofibrationSpec, verify_cofibration

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
    return (MorphSpec("translate", "panel"), MorphSpec("uniform_scale", "panel"))


def _object_count_hypothesis() -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id="two_principal_objects",
        description="Positive panels contain two principal objects; negatives contain one.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("score", LegCall("object_count", ("scene",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
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
    )


def _triangle_proxy_hypothesis() -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id="triangle_by_aspect_proxy",
        description="The figure contains a triangle attached to a square.",
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


def test_triangle_semantics_require_primitive_witness_path():
    cone = compile_hypothesis(_triangle_witness_hypothesis(), default_registry())
    assert cone.node_types["contour"] == "ContourWitness"
    assert cone.node_types["polygon"] == "PolygonWitness"
    assert cone.node_types["triangle"] == "TriangleWitness"
    assert "triangle" in cone.node_dependencies["score"]


def test_problem_05_style_fish_proxy_is_not_semantic_clean():
    hyp = SemanticHypothesis(
        version="0.1",
        hypothesis_id="symmetric_fish_by_area",
        description="The positive figure is a symmetric fish-like object.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("score", LegCall("largest_area", ("scene",))),
        )),
        score_node="score",
        order="high_positive",
        preservation_morphisms=_morphism(),
        semantic_requirements=("fish-like",),
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
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("main", LegCall("select_largest", ("scene",))),
            DiagramEdge("score", LegCall("closure_ratio", ("main",))),
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
    )
    result = verify_hypothesis(hyp, default_registry(), _problem_two_objects_vs_one())
    assert result.n_examples == 12
    assert result.support_errors == 12
    assert result.loo_errors == 12


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


def test_cofibration_preservation_check_is_mechanical():
    spec = CofibrationSpec(
        name="source_to_target",
        source_type="dict",
        target_type="dict",
        preserved_fields=("id",),
        interface_fields=("contact",),
        added_fields=("appendage",),
        attachment_leg="attach",
    )
    assert verify_cofibration(
        {"id": "body"},
        {"id": "body", "contact": "neck", "appendage": "wing"},
        spec,
    ).ok
    failed = verify_cofibration(
        {"id": "body"},
        {"id": "other", "contact": "neck", "appendage": "wing"},
        spec,
    )
    assert not failed.ok
    assert failed.first_failed == "witness_preservation"


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
