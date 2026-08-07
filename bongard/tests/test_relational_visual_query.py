from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw

from bongard.evidence import Disposition
from bongard.loop_scene_witnesses import extract_loop_scene_witnesses
from bongard.relational_visual_query import (
    EdgeObliquenessClause,
    PointContactClause,
    Rational,
    RelationalVisualQuery,
    ScenarioQueryResult,
    SideCountClause,
    _edge_obliqueness,
    _existential_disposition,
    _side_count,
    enumerate_factorized_shape_ratio_queries,
    evaluate_relational_query,
    exact_support_separators,
    verify_relational_query_result,
)


def _panel(*, triangle_radius: int, quadrilateral_radius: int) -> bytes:
    image = Image.new("RGB", (160, 160), "white")
    draw = ImageDraw.Draw(image)
    tc = (42, 82)
    triangle = [
        (tc[0], tc[1] - triangle_radius),
        (tc[0] - triangle_radius, tc[1] + triangle_radius),
        (tc[0] + triangle_radius, tc[1] + triangle_radius),
    ]
    qc = (112, 82)
    q = quadrilateral_radius
    quadrilateral = [
        (qc[0] - q, qc[1] - q),
        (qc[0] + q, qc[1] - q + 5),
        (qc[0] + q - 4, qc[1] + q),
        (qc[0] - q + 3, qc[1] + q - 5),
    ]
    draw.line(triangle + [triangle[0]], fill="black", width=4, joint="curve")
    draw.line(
        quadrilateral + [quadrilateral[0]], fill="black", width=4, joint="curve"
    )
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _query() -> RelationalVisualQuery:
    return RelationalVisualQuery.factorized_shape_ratio(
        numerator_side_count=3,
        denominator_side_count=4,
        ratio=Rational(1, 8),
    )


def _touching_panel() -> bytes:
    image = Image.new("RGB", (160, 144), "white")
    draw = ImageDraw.Draw(image)
    triangle = [(8, 22), (8, 122), (65, 72)]
    quadrilateral = [(65, 72), (110, 8), (154, 72), (110, 136)]
    draw.line(triangle + [triangle[0]], fill="black", width=2, joint="curve")
    draw.line(
        quadrilateral + [quadrilateral[0]], fill="black", width=2, joint="curve"
    )
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def _uncertain_heptagon_and_quadrilateral_panel() -> bytes:
    """Seven literal input vertices that the frozen variants resolve as 5/6."""

    heptagon = [
        (89.154, 82.813),
        (64.915, 93.453),
        (32.571, 90.701),
        (31.497, 57.636),
        (47.321, 31.951),
        (80.031, 27.885),
        (95.945, 55.658),
    ]
    quadrilateral = [(130, 25), (225, 35), (215, 130), (125, 120)]
    image = Image.new("RGB", (250, 160), "white")
    draw = ImageDraw.Draw(image)
    draw.line(heptagon + [heptagon[0]], fill="black", width=3, joint="curve")
    draw.line(
        quadrilateral + [quadrilateral[0]],
        fill="black",
        width=3,
        joint="curve",
    )
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    return output.getvalue()


def test_same_binding_query_accepts_small_triangle_large_quadrilateral() -> None:
    packet = extract_loop_scene_witnesses(
        _panel(triangle_radius=11, quadrilateral_radius=34)
    )
    query = _query()
    result = evaluate_relational_query(query, packet)

    assert result.disposition is Disposition.PRESENT
    assert all(item.disposition is Disposition.PRESENT for item in result.scenarios)
    assert RelationalVisualQuery.from_data(query.to_data()) == query
    assert verify_relational_query_result(result, query, packet) is result
    assert "not" not in str(query.to_data()).lower()


def test_role_reversal_and_similar_size_are_certified_absent() -> None:
    reversed_packet = extract_loop_scene_witnesses(
        _panel(triangle_radius=34, quadrilateral_radius=11)
    )
    similar_packet = extract_loop_scene_witnesses(
        _panel(triangle_radius=25, quadrilateral_radius=27)
    )

    assert evaluate_relational_query(_query(), reversed_packet).disposition is (
        Disposition.CERTIFIED_ABSENT
    )
    assert evaluate_relational_query(_query(), similar_packet).disposition is (
        Disposition.CERTIFIED_ABSENT
    )


def test_no_roles_is_exhaustive_absence_not_indeterminate() -> None:
    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).line((15, 64, 110, 64), fill="black", width=4)
    output = BytesIO()
    image.save(output, format="PNG", optimize=False)
    result = evaluate_relational_query(
        _query(), extract_loop_scene_witnesses(output.getvalue())
    )

    assert result.disposition is Disposition.CERTIFIED_ABSENT
    assert all(not item.bindings for item in result.scenarios)


def test_unresolved_role_domain_prevents_vacuous_existence_absence() -> None:
    assert _existential_disposition((), ()) is Disposition.CERTIFIED_ABSENT
    assert _existential_disposition(
        (), (Disposition.INDETERMINATE,)
    ) is Disposition.INDETERMINATE
    assert _existential_disposition(
        (), (Disposition.PRESENT, Disposition.INDETERMINATE)
    ) is Disposition.INDETERMINATE
    assert _existential_disposition(
        (), (Disposition.PRESENT, Disposition.ERROR)
    ) is Disposition.ERROR

    empty_unresolved = ScenarioQueryResult(
        scenario_id="threshold032.raw",
        role_domain=(("loop-00000000", Disposition.INDETERMINATE),),
        bindings=(),
        disposition=Disposition.INDETERMINATE,
        reason_code="unresolved_binding",
    )
    one_present_one_unresolved = ScenarioQueryResult(
        scenario_id="threshold032.raw",
        role_domain=(
            ("loop-00000000", Disposition.PRESENT),
            ("loop-00000001", Disposition.INDETERMINATE),
        ),
        bindings=(),
        disposition=Disposition.INDETERMINATE,
        reason_code="unresolved_binding",
    )
    assert not empty_unresolved.bindings
    assert not one_present_one_unresolved.bindings

    present_result = evaluate_relational_query(
        _query(),
        extract_loop_scene_witnesses(
            _panel(triangle_radius=11, quadrilateral_radius=34)
        ),
    )
    present_binding = next(
        binding
        for scenario in present_result.scenarios
        for binding in scenario.bindings
        if binding.disposition is Disposition.PRESENT
    )
    assert _existential_disposition(
        (present_binding,), (Disposition.ERROR,)
    ) is Disposition.PRESENT


def test_indeterminate_geometry_cannot_become_threshold_absence() -> None:
    packet = extract_loop_scene_witnesses(
        _uncertain_heptagon_and_quadrilateral_panel()
    )
    for scenario in packet.scenarios:
        uncertain = scenario.loops[0]
        assert uncertain.polygon.disposition is Disposition.INDETERMINATE
        assert uncertain.polygon.side_count is not None
        assert _side_count(
            SideCountClause("clause-00", "role-00", 7), uncertain
        ).disposition is Disposition.INDETERMINATE
        assert _edge_obliqueness(
            EdgeObliquenessClause("clause-00", "role-00", 10_000),
            uncertain,
        ).disposition is Disposition.INDETERMINATE

    query = RelationalVisualQuery.factorized_shape_ratio(
        numerator_side_count=7,
        denominator_side_count=4,
        ratio=Rational(1, 2),
    )
    result = evaluate_relational_query(query, packet)
    assert result.disposition is Disposition.INDETERMINATE
    assert all(
        item.disposition is Disposition.INDETERMINATE for item in result.scenarios
    )


def test_support_search_is_positive_only_and_finds_factorized_family() -> None:
    positives = [
        extract_loop_scene_witnesses(
            _panel(triangle_radius=radius, quadrilateral_radius=34)
        )
        for radius in (9, 10, 11)
    ]
    negatives = [
        extract_loop_scene_witnesses(
            _panel(triangle_radius=34, quadrilateral_radius=radius)
        )
        for radius in (9, 10, 11)
    ]
    separators = exact_support_separators(positives, negatives)

    assert separators
    assert all(
        query.clauses[0].count == 3 and query.clauses[1].count == 4
        for query in separators
    )
    assert all("not" not in str(query.to_data()).lower() for query in separators)


def test_point_contact_clause_uses_the_same_bound_pair() -> None:
    packet = extract_loop_scene_witnesses(_touching_panel())
    query = RelationalVisualQuery.factorized_shape_ratio(
        numerator_side_count=3,
        denominator_side_count=4,
        ratio=Rational(1, 2),
        require_point_contact=True,
    )
    result = evaluate_relational_query(query, packet)

    assert result.disposition is Disposition.PRESENT
    assert all(
        scenario.disposition is Disposition.PRESENT for scenario in result.scenarios
    )
    for scenario in result.scenarios:
        witnesses = [
            binding
            for binding in scenario.bindings
            if binding.disposition is Disposition.PRESENT
        ]
        assert len(witnesses) == 1
        assert witnesses[0].bindings == (
            ("role-00", "loop-00000000"),
            ("role-01", "loop-00000001"),
        )


def test_finite_search_enumerates_optional_obliqueness_and_contact() -> None:
    queries = enumerate_factorized_shape_ratio_queries()

    assert len(queries) == 6 * 6 * 7 * 5 * 2
    assert any(
        any(isinstance(clause, EdgeObliquenessClause) for clause in query.clauses)
        for query in queries
    )
    assert any(
        any(isinstance(clause, PointContactClause) for clause in query.clauses)
        for query in queries
    )
