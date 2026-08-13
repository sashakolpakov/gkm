from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from bongard import panel_action_count_ordered_path_inversion as subject
from bongard import panel_action_count_synthetic_identifiability as fixtures


EXPECTED_SOURCE_SHA256 = "36e2c5ad738e7d56dda04f9bd97dd0f39f095862d216c1c6b6a01469a604f0e5"


def _png_blank() -> bytes:
    output = BytesIO()
    Image.new("L", (64, 64), 255).save(output, format="PNG")
    return output.getvalue()


def test_png_only_inversion_recovers_canonical_visible_line_and_arc() -> None:
    rows = fixtures.build_identifiability_counterfactuals()
    outcomes = tuple(subject.invert_png(row.panel.png_bytes) for row in rows)
    assert tuple(outcome.candidate_pairs for outcome in outcomes) == (
        ((1, 0),), ((1, 0),), ((0, 1),), ((0, 1),),
    )
    assert all(outcome.disposition == "IDENTIFIED" for outcome in outcomes)
    assert all(len(outcome.paths) == 1 for outcome in outcomes)
    assert all(outcome.skeleton_pixel_count > 0 for outcome in outcomes)


def test_hidden_generator_boundaries_do_not_change_visible_inversion() -> None:
    one_line, two_lines, one_arc, two_arcs = fixtures.build_identifiability_counterfactuals()
    assert one_line.panel.declared_pair != two_lines.panel.declared_pair
    assert subject.invert_png(one_line.panel.png_bytes).candidate_pairs == subject.invert_png(
        two_lines.panel.png_bytes
    ).candidate_pairs
    assert one_arc.panel.declared_pair != two_arcs.panel.declared_pair
    assert subject.invert_png(one_arc.panel.png_bytes).candidate_pairs == subject.invert_png(
        two_arcs.panel.png_bytes
    ).candidate_pairs


def test_d4_nuisance_preserves_a_line_visible_count() -> None:
    program = fixtures.Program(
        "d4",
        (
            fixtures.LineAction(
                "line", fixtures.Point(128, 512), fixtures.Point(896, 512)
            ),
        ),
    )
    for d4 in fixtures.D4_NAMES:
        panel = fixtures.render_program(program, fixtures.Nuisance(d4, 3, 1000))
        outcome = subject.invert_png(panel.png_bytes)
        assert outcome.candidate_pairs == ((1, 0),)


def test_path_graph_retains_order_incidence_and_fit_ranges() -> None:
    panel = fixtures.render_program(
        fixtures.Program(
            "separate",
            (
                fixtures.LineAction(
                    "first", fixtures.Point(100, 200), fixtures.Point(400, 200)
                ),
                fixtures.LineAction(
                    "second", fixtures.Point(600, 800), fixtures.Point(900, 800)
                ),
            ),
        )
    )
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.candidate_pairs == ((2, 0),)
    assert len(outcome.paths) == 2
    assert {path.component_id for path in outcome.paths} == {0, 1}
    assert all(path.start_degree == path.end_degree == 1 for path in outcome.paths)
    assert all(fit.segment_start < fit.segment_end for fit in outcome.hypotheses[0].fits)


def test_malformed_empty_and_wrong_tolerance_fail_closed() -> None:
    with pytest.raises(subject.OrderedPathInversionError, match="not issued"):
        subject.invert_png(b"not-png")
    with pytest.raises(subject.OrderedPathInversionError, match="not issued"):
        subject.invert_png(_png_blank())
    panel = fixtures.build_identifiability_counterfactuals()[0].panel
    with pytest.raises(subject.OrderedPathInversionError, match="tolerance"):
        subject.invert_png(panel.png_bytes, line_tolerance=1)  # type: ignore[arg-type]


def test_genuine_junction_is_gap_not_a_silently_dropped_branch() -> None:
    panel = fixtures.render_program(
        fixtures.Program(
            "short-t",
            (
                fixtures.LineAction(
                    "stem", fixtures.Point(512, 160), fixtures.Point(512, 864)
                ),
                fixtures.LineAction(
                    "branch", fixtures.Point(512, 512), fixtures.Point(720, 512)
                ),
            ),
        )
    )
    assert panel.canonical_visible_pair is None
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.disposition == "GAP"
    assert outcome.reason == "junction_graph_requires_global_path_cover"
    assert outcome.candidate_pairs == ()
    assert any(path.start_degree >= 3 or path.end_degree >= 3 for path in outcome.paths)


def test_one_pixel_residual_at_degree_three_is_never_backbone_suppressed() -> None:
    """Regression for a short rendered branch hidden by geodesic coverage."""

    panel = fixtures.render_program(
        fixtures.Program(
            "short-rendered-t",
            (
                fixtures.LineAction(
                    "stem", fixtures.Point(512, 160), fixtures.Point(512, 864)
                ),
                fixtures.LineAction(
                    "short-branch",
                    fixtures.Point(553, 453),
                    fixtures.Point(512, 512),
                ),
            ),
        )
    )
    assert panel.canonical_visible_pair is None
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.disposition == "GAP"
    assert outcome.reason == "junction_graph_requires_global_path_cover"
    assert any(path.start_degree >= 3 or path.end_degree >= 3 for path in outcome.paths)


def test_visible_crossbar_erased_by_thinning_returns_residual_gap() -> None:
    panel = fixtures.render_program(
        fixtures.Program(
            "erased-crossbar",
            (
                fixtures.LineAction(
                    "stem", fixtures.Point(512, 160), fixtures.Point(512, 864)
                ),
                fixtures.LineAction(
                    "crossbar",
                    fixtures.Point(480, 512),
                    fixtures.Point(544, 512),
                ),
            ),
        )
    )
    assert panel.canonical_visible_pair is None
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.disposition == "GAP"
    assert outcome.reason == (
        "foreground_residual_exceeds_single_path_stroke_envelope"
    )
    assert outcome.candidate_pairs == ()


def test_endpoint_branch_exactly_equivalent_to_one_line_has_one_visible_target() -> None:
    case = next(
        case for case in fixtures.ambiguity_cases()
        if case.case_id == "endpoint-branch-vs-one-raster-equivalent-line"
    )
    left = fixtures.render_program(case.left)
    right = fixtures.render_program(case.right)
    assert left.png_bytes == right.png_bytes
    assert left.png_sha256 == (
        "sha256:30663fa5107abdb6c7e2fa7b09361a59d7de4d46c6775ed5b6951527ca2b4331"
    )
    assert left.canonical_visible_pair == right.canonical_visible_pair == fixtures.CountPair(1, 0)
    outcome = subject.invert_png(left.png_bytes)
    assert outcome.disposition == "IDENTIFIED"
    assert outcome.candidate_pairs == ((1, 0),)


def test_line_chain_exactly_equivalent_to_one_arc_has_one_visible_target() -> None:
    case = next(
        case for case in fixtures.ambiguity_cases()
        if case.case_id == "three-line-chain-vs-one-raster-equivalent-arc"
    )
    line_chain = fixtures.render_program(case.left)
    one_arc = fixtures.render_program(case.right)
    assert line_chain.png_bytes == one_arc.png_bytes
    assert line_chain.png_sha256 == (
        "sha256:b5bd0067160223fe3aca48471e63df7d5ded95066ab2ee446470c68e99be1e63"
    )
    assert line_chain.canonical_visible_pair == one_arc.canonical_visible_pair == (
        fixtures.CountPair(0, 1)
    )
    outcome = subject.invert_png(line_chain.png_bytes)
    assert outcome.disposition == "IDENTIFIED"
    assert outcome.candidate_pairs == ((0, 1),)


@pytest.mark.parametrize(
    "case_id,wanted",
    (
        ("endpoint-branch-alias-with-disconnected-context", (2, 0)),
        ("line-chain-arc-alias-with-disconnected-context", (1, 1)),
    ),
)
def test_exact_aliases_remain_normalized_under_disconnected_context(
    case_id: str, wanted: tuple[int, int]
) -> None:
    case = next(
        case for case in fixtures.ambiguity_cases()
        if case.case_id == case_id
    )
    left = fixtures.render_program(case.left)
    right = fixtures.render_program(case.right)
    assert left.png_bytes == right.png_bytes
    assert left.canonical_visible_pair == right.canonical_visible_pair == (
        fixtures.CountPair(*wanted)
    )
    outcome = subject.invert_png(left.png_bytes)
    assert outcome.disposition == "IDENTIFIED"
    assert outcome.candidate_pairs == (wanted,)


@pytest.mark.parametrize(
    "case_id",
    (
        "endpoint-branch-alias-with-touching-context",
        "line-chain-arc-alias-with-touching-context",
    ),
)
def test_exact_aliases_with_touching_context_have_unresolved_visible_target(
    case_id: str,
) -> None:
    case = next(
        case for case in fixtures.ambiguity_cases()
        if case.case_id == case_id
    )
    left = fixtures.render_program(case.left)
    right = fixtures.render_program(case.right)
    assert left.png_bytes == right.png_bytes
    assert left.canonical_visible_pair is None
    assert right.canonical_visible_pair is None

    outcome = subject.invert_png(left.png_bytes)
    assert outcome.disposition in ("AMBIGUOUS", "GAP")
    assert outcome.disposition != "IDENTIFIED"
    if outcome.disposition == "GAP":
        assert outcome.candidate_pairs == ()
        assert outcome.reason is not None


@pytest.mark.parametrize(
    "angle_degrees,length",
    ((105, 160), (110, 128), (110, 160)),
)
def test_visible_endpoint_branch_never_becomes_a_false_one_primitive_set(
    angle_degrees: int, length: int
) -> None:
    import math

    radians = math.radians(angle_degrees)
    dx = round(math.cos(radians) * length / 2)
    dy = round(math.sin(radians) * length / 2)
    panel = fixtures.render_program(
        fixtures.Program(
            f"endpoint-branch-{angle_degrees}-{length}",
            (
                fixtures.LineAction(
                    "stem", fixtures.Point(512, 160), fixtures.Point(512, 864)
                ),
                fixtures.LineAction(
                    "branch",
                    fixtures.Point(512 - dx, 160 - dy),
                    fixtures.Point(512 + dx, 160 + dy),
                ),
            ),
        )
    )
    assert panel.canonical_visible_pair is None
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.disposition in ("AMBIGUOUS", "GAP")
    assert outcome.disposition != "IDENTIFIED"


@pytest.mark.parametrize(
    "program",
    (
        fixtures.Program(
            "shallow-corner",
            (
                fixtures.LineAction(
                    "first", fixtures.Point(192, 512), fixtures.Point(512, 512)
                ),
                fixtures.LineAction(
                    "second", fixtures.Point(512, 512), fixtures.Point(632, 556)
                ),
            ),
        ),
        fixtures.Program(
            "offset-endpoint-branch",
            (
                fixtures.LineAction(
                    "stem", fixtures.Point(512, 160), fixtures.Point(512, 864)
                ),
                fixtures.LineAction(
                    "branch", fixtures.Point(555, 69), fixtures.Point(501, 219)
                ),
            ),
        ),
        fixtures.Program(
            "moderate-corner",
            (
                fixtures.LineAction(
                    "first", fixtures.Point(192, 512), fixtures.Point(512, 512)
                ),
                fixtures.LineAction(
                    "second", fixtures.Point(512, 512), fixtures.Point(693, 693)
                ),
            ),
        ),
    ),
)
def test_shallow_line_junction_never_becomes_a_false_single_arc(
    program: fixtures.Program,
) -> None:
    panel = fixtures.render_program(program)
    assert panel.canonical_visible_pair is None
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.disposition == "GAP"
    assert outcome.candidate_pairs == ()


@pytest.mark.parametrize("segment_count", range(4, 10))
def test_polygonal_semicircle_is_not_accepted_as_one_exact_arc(
    segment_count: int,
) -> None:
    import math

    points = tuple(
        fixtures.Point(
            512 + round(320 * math.cos(math.pi - math.pi * index / segment_count)),
            512 - round(320 * math.sin(math.pi - math.pi * index / segment_count)),
        )
        for index in range(segment_count + 1)
    )
    panel = fixtures.render_program(
        fixtures.Program(
            "polygonal-semicircle",
            tuple(
                fixtures.LineAction(str(index), start, end)
                for index, (start, end) in enumerate(zip(points, points[1:]))
            ),
        )
    )
    assert panel.canonical_visible_pair is None
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.disposition == "GAP"
    assert outcome.reason == (
        "raw_foreground_not_explained_by_single_circular_arc"
    )


def test_legal_shallow_arc_is_never_relabelled_as_lines() -> None:
    panel = fixtures.render_program(
        fixtures.Program(
            "legal-shallow-arc",
            (
                fixtures.ArcAction(
                    "arc",
                    fixtures.Point(432, 512),
                    fixtures.Point(512, 480),
                    fixtures.Point(592, 512),
                ),
            ),
        )
    )
    assert panel.canonical_visible_pair == fixtures.CountPair(0, 1)
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.disposition == "IDENTIFIED"
    assert outcome.candidate_pairs == ((0, 1),)


@pytest.mark.parametrize("angle_degrees,length", ((45, 64), (60, 160), (75, 160)))
def test_diagonal_crossbar_erased_by_thinning_never_becomes_one_line(
    angle_degrees: int, length: int
) -> None:
    import math

    radians = math.radians(angle_degrees)
    dx = round(math.cos(radians) * length / 2)
    dy = round(math.sin(radians) * length / 2)
    panel = fixtures.render_program(
        fixtures.Program(
            f"erased-diagonal-{angle_degrees}-{length}",
            (
                fixtures.LineAction(
                    "stem", fixtures.Point(512, 160), fixtures.Point(512, 864)
                ),
                fixtures.LineAction(
                    "crossbar",
                    fixtures.Point(512 - dx, 512 - dy),
                    fixtures.Point(512 + dx, 512 + dy),
                ),
            ),
        )
    )
    assert panel.canonical_visible_pair is None
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.disposition == "GAP"
    assert outcome.candidate_pairs == ()


def test_every_bounded_single_line_panel_survives_raw_stroke_explanation() -> None:
    panels = tuple(
        sample.panel
        for sample in fixtures.build_balanced_corpus()
        if sample.panel.canonical_visible_pair == fixtures.CountPair(1, 0)
    )
    assert len(panels) == len(fixtures.CARRIER_FAMILIES) * len(
        fixtures.default_nuisances()
    )
    outcomes = tuple(subject.invert_png(panel.png_bytes) for panel in panels)
    assert all(
        outcome.disposition == "GAP" or (1, 0) in outcome.candidate_pairs
        for outcome in outcomes
    )
    assert not any(
        outcome.reason == "raw_foreground_not_explained_by_single_straight_stroke"
        for outcome in outcomes
    )


def test_every_bounded_single_arc_panel_is_identified_or_explicitly_gapped() -> None:
    panels = tuple(
        fixtures.render_program(
            fixtures._carrier_program(fixtures.CountPair(0, 1), family),
            nuisance,
        )
        for family in fixtures.CARRIER_FAMILIES
        for nuisance in fixtures.default_nuisances()
    )
    assert len(panels) == 20
    outcomes = tuple(subject.invert_png(panel.png_bytes) for panel in panels)
    assert all(
        outcome.disposition == "GAP" or outcome.candidate_pairs == ((0, 1),)
        for outcome in outcomes
    )
    assert not any(
        outcome.disposition == "IDENTIFIED"
        and outcome.candidate_pairs != ((0, 1),)
        for outcome in outcomes
    )


def test_exported_path_and_outcome_reject_missing_incidence_and_unknown_fits() -> None:
    with pytest.raises(subject.OrderedPathInversionError, match="endpoint degrees"):
        subject.OrderedGraphPath(0, 0, 0, 1, False, ((0, 0), (0, 1)))
    with pytest.raises(subject.OrderedPathInversionError, match="endpoint degrees"):
        subject.OrderedGraphPath(0, 0, 2, 1, False, ((0, 0), (0, 1)))

    path = subject.OrderedGraphPath(0, 0, 1, 1, False, ((0, 0), (0, 1)))
    fit = subject.PrimitiveFit(1, 0, 1, "line", 0.0, 1)
    hypothesis = subject.ProgramHypothesis(1, 0, 0.0, (fit,))
    with pytest.raises(subject.OrderedPathInversionError, match="fit path ids"):
        subject.InversionOutcome("IDENTIFIED", (path,), (hypothesis,), None, 2, 2)


def test_pair_keyed_hypothesis_composition_is_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = tuple(
        subject.OrderedGraphPath(index, index, 1, 1, False, ((0, index), (1, index)))
        for index in range(9)
    )
    monkeypatch.setattr(subject, "_trace_paths", lambda _skeleton: paths)

    def alternatives(
        path: subject.OrderedGraphPath, **_kwargs: object
    ) -> tuple[tuple[subject.PrimitiveFit, ...], ...]:
        return (
            (subject.PrimitiveFit(path.path_id, 0, 1, "line", 0.1, 1),),
            (subject.PrimitiveFit(path.path_id, 0, 1, "arc", 0.1, 1),),
        )

    monkeypatch.setattr(subject, "_fit_path", alternatives)
    panel = fixtures.render_program(
        fixtures.Program(
            "hypothesis-inventory",
            (
                fixtures.LineAction(
                    "first", fixtures.Point(128, 512), fixtures.Point(512, 512)
                ),
                fixtures.LineAction(
                    "second", fixtures.Point(512, 512), fixtures.Point(512, 896)
                ),
            ),
        )
    )
    assert panel.canonical_visible_pair is None
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.disposition == "AMBIGUOUS"
    assert outcome.reason is None
    assert outcome.candidate_pairs == tuple(
        (straight, 9 - straight) for straight in range(10)
    )
    assert len(outcome.hypotheses) == 10


def test_unresolved_raster_never_issues_a_singleton_from_one_dp_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = fixtures.render_program(
        fixtures.Program(
            "unresolved-singleton-guard",
            (
                fixtures.LineAction(
                    "first", fixtures.Point(128, 512), fixtures.Point(512, 512)
                ),
                fixtures.LineAction(
                    "second", fixtures.Point(512, 512), fixtures.Point(512, 896)
                ),
            ),
        )
    )
    assert panel.canonical_visible_pair is None
    path = subject.OrderedGraphPath(
        0, 0, 1, 1, False, ((32, 8), (32, 9), (32, 10))
    )
    monkeypatch.setattr(subject, "_trace_paths", lambda _skeleton: (path,))
    monkeypatch.setattr(
        subject,
        "_fit_path",
        lambda _path, **_kwargs: ((
            subject.PrimitiveFit(0, 0, 1, "line", 0.1, 1),
            subject.PrimitiveFit(0, 1, 2, "line", 0.1, 1),
        ),),
    )
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.disposition == "GAP"
    assert outcome.reason == "unresolved_raster_component_cannot_issue_singleton"
    assert outcome.candidate_pairs == ()


def test_exact_component_normal_form_cannot_be_bypassed_by_multifit_dp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = fixtures.render_program(
        fixtures.Program(
            "normal-form-mismatch",
            (
                fixtures.LineAction(
                    "line", fixtures.Point(128, 512), fixtures.Point(896, 512)
                ),
            ),
        )
    )

    def only_two_lines(path: subject.OrderedGraphPath, **_kwargs: object):
        midpoint = max(1, (len(path.pixels_yx) - 1) // 2)
        return ((
            subject.PrimitiveFit(path.path_id, 0, midpoint, "line", 0.1, 1),
            subject.PrimitiveFit(
                path.path_id,
                midpoint,
                len(path.pixels_yx) - 1,
                "line",
                0.1,
                1,
            ),
        ),)

    monkeypatch.setattr(subject, "_fit_path", only_two_lines)
    outcome = subject.invert_png(panel.png_bytes)
    assert outcome.disposition == "GAP"
    assert outcome.reason == (
        "exact_raster_normal_form_missing_from_path_hypotheses"
    )
    assert outcome.candidate_pairs == ()


def test_path_partition_preserves_every_admissible_minimum_complexity_pair() -> None:
    """No greedy split erases pairs admitted by the conservative arc grammar."""

    pixels = (
        (32, 32), (33, 32), (34, 32), (35, 32), (36, 32), (37, 32),
        (38, 33), (39, 34), (40, 35), (41, 34), (42, 33),
    )
    path = subject.OrderedGraphPath(0, 0, 1, 1, False, pixels)
    partitions = subject._fit_path(
        path, line_tolerance=0.55, arc_tolerance=0.70
    )
    pairs = {
        (
            sum(fit.kind == "line" for fit in partition),
            sum(fit.kind == "arc" for fit in partition),
        )
        for partition in partitions
    }

    assert pairs == {(0, 2), (1, 1), (2, 0)}
    assert len(partitions) == len(pairs) == 3
    assert all(len(partition) == 2 for partition in partitions)


def test_solver_surface_contains_no_labels_carrier_or_official_loader() -> None:
    assert not hasattr(subject, "main")
    assert tuple(subject.invert_png.__annotations__) == (
        "png_bytes", "line_tolerance", "arc_tolerance", "return"
    )
    source = open(subject.__file__, encoding="utf-8").read()  # noqa: PTH123
    assert "downloads/" not in source
    assert "declared_pair" not in source
    assert "carrier_family" not in source
    assert subject.source_sha256() == EXPECTED_SOURCE_SHA256
