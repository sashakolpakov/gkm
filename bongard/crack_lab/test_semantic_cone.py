import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import Problem, _draw_polyline, trace_shape
from semantic_compiler import CompileError, compile_hypothesis
from semantic_ir import DiagramEdge, DiagramSpec, LegCall, MorphSpec, SemanticHypothesis
from semantic_legs import default_registry
import semantic_legs as L
import semantic_artifacts as SA
from semantic_verifier import verify_hypothesis
from semantic_selection import CandidateEvaluation, ComplexityBreakdown, RiskVector, Track, pareto_frontier
from cofibrations import CofibrationSpec, verify_cofibration
from cofibered_proposer import build_prompt, hypotheses_from_tool_input
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


def test_polygon_side_counts_from_strokes():
    tri = [(30, 30), (90, 30), (60, 80), (30, 30)]
    sq = [(30, 30), (90, 30), (90, 90), (30, 90), (30, 30)]
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    circle = np.stack([64 + 30 * np.cos(theta), 64 + 30 * np.sin(theta)], axis=1)

    def poly_of(polyline):
        scene = L.parse_scene(_panel_from_polylines([polyline]))
        return L.fit_polygon(L.extract_contours(L.select_largest(scene)))

    tri_poly, sq_poly, circle_poly = poly_of(tri), poly_of(sq), poly_of(circle)
    assert tri_poly.side_count == 3
    assert sq_poly.side_count == 4
    assert circle_poly.side_count >= 6
    assert L.classify_triangle(tri_poly).confidence > 0.0
    assert L.classify_quadrilateral(sq_poly).confidence > 0.0
    for bad, cls in ((sq_poly, L.classify_triangle),
                     (circle_poly, L.classify_triangle)):
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


def test_structured_proposal_parsing_is_per_item_tolerant():
    good = {
        "hypothesis_id": "h1",
        "description": "The principal object is an open curve.",
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
        "preservation_morphisms": [{"name": "translate"}],
    }
    bad = {"description": "no hypothesis_id or score_node"}
    hyps, err = hypotheses_from_tool_input({"hypotheses": [good, bad]})
    assert len(hyps) == 1
    assert hyps[0].hypothesis_id == "h1"
    assert "hypothesis[1]" in err
    result = verify_hypothesis(hyps[0], default_registry(),
                               _problem_two_objects_vs_one())
    assert result.compile_error == ""


def test_prompt_leg_list_is_generated_from_registry():
    prompt = build_prompt("problem_00")
    for name in default_registry().names():
        assert f"- {name}:" in prompt
    # no black-box composite concept legs are advertised
    for forbidden in ("bird", "fish", "lamp", "pinwheel", "prototype"):
        assert forbidden not in prompt.lower()


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

        old_lab = SA.LAB_DIR
        SA.LAB_DIR = td
        try:
            dest = SA.snapshot_wip("unittest", out, "problem_00")
            assert os.path.exists(
                os.path.join(dest, "problem_00_round00_proposal.txt"))
            art = SA.promote(
                "unittest", out, {"records": []},
                {"problem_00": {"solved": True, "concept": "harness-only",
                                "status": "SOLVED_SEMANTIC_PURE", "rule": "r"}},
                [])
            assert os.path.exists(os.path.join(art, "results.json"))
            assert os.path.exists(os.path.join(art, "README.md"))
            assert os.path.exists(os.path.join(art, "promoted_cones.json"))
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
