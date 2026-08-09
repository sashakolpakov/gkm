"""Synthetic tests for hierarchical macro-action versus micro-ink geometry."""

from __future__ import annotations

from copy import deepcopy

import pytest

from bongard.panel_feature_observation import EngineeringFeatureDisposition
from bongard.panel_hierarchical_action_geometry import (
    DerivedMacroSpanKind,
    GeometryDerivationStatus,
    GeometryEvidenceProvenance,
    GeometryTraceIssue,
    Grid16Interval,
    HierarchicalActionGeometryError,
    HierarchicalActionGeometryEvidence,
    HierarchicalActionGeometryReplay,
    MacroActionPrimitive,
    MacroActionSpan,
    MacroActionTrace,
    MicroTextureEvidence,
    MicroTexturePrimitive,
    MicroTexturePrimitiveKind,
    TraceResolution,
    UncertainGrid16Point,
    cold_replay_hierarchical_action_geometry,
    derive_macro_convexity,
    derive_macro_straight_span_count,
    evaluate_macro_convexity,
    evaluate_macro_straight_span_count,
    evaluate_positive_macro_conjunction,
)
from bongard.panel_soft_ontology import ClosedCount, ConvexityKind


def _p(x: int, y: int) -> UncertainGrid16Point:
    return UncertainGrid16Point.exact(x, y)


def _line(a: UncertainGrid16Point, b: UncertainGrid16Point) -> MacroActionSpan:
    return MacroActionSpan(
        TraceResolution.COMPLETE,
        MacroActionPrimitive.LINE,
        (a, b),
        None,
    )


def _arc(
    a: UncertainGrid16Point,
    control: UncertainGrid16Point,
    b: UncertainGrid16Point,
) -> MacroActionSpan:
    return MacroActionSpan(
        TraceResolution.COMPLETE,
        MacroActionPrimitive.ARC,
        (a, control, b),
        None,
    )


def _trace(vertices: tuple[UncertainGrid16Point, ...]) -> MacroActionTrace:
    return MacroActionTrace.complete(
        tuple(
            _line(point, vertices[(index + 1) % len(vertices)])
            for index, point in enumerate(vertices)
        )
    )


def _provenance() -> GeometryEvidenceProvenance:
    return GeometryEvidenceProvenance(
        panel_png_digest="1" * 64,
        panel_png_byte_count=100,
        observer_contract_digest="2" * 64,
        measurement_protocol_digest="3" * 64,
        observation_receipt_digest="4" * 64,
    )


def _evidence(
    trace: MacroActionTrace,
    micro: MicroTextureEvidence | None = None,
) -> HierarchicalActionGeometryEvidence:
    return HierarchicalActionGeometryEvidence.create(
        _provenance(),
        trace,
        MicroTextureEvidence.complete() if micro is None else micro,
    )


SQUARE = (_p(2, 2), _p(13, 2), _p(13, 13), _p(2, 13))
CONCAVE_FOUR = (_p(2, 2), _p(13, 2), _p(6, 6), _p(2, 13))
CONVEX_FIVE = (_p(3, 2), _p(12, 2), _p(14, 8), _p(8, 14), _p(2, 9))


def test_four_line_convex_conjunction_and_heterogeneous_negatives() -> None:
    positive = _evidence(_trace(SQUARE))
    negative_not_convex = _evidence(_trace(CONCAVE_FOUR))
    negative_not_four = _evidence(_trace(CONVEX_FIVE))

    assert evaluate_positive_macro_conjunction(
        positive,
        convexity=ConvexityKind.CONVEX_CLOSED_BOUNDARY,
        straight_span_count=ClosedCount.FOUR,
    ) is EngineeringFeatureDisposition.MATCH

    assert evaluate_macro_convexity(
        negative_not_convex, ConvexityKind.CONVEX_CLOSED_BOUNDARY
    ) is EngineeringFeatureDisposition.NONMATCH
    assert evaluate_macro_straight_span_count(
        negative_not_convex, ClosedCount.FOUR
    ) is EngineeringFeatureDisposition.MATCH
    assert evaluate_positive_macro_conjunction(
        negative_not_convex,
        convexity=ConvexityKind.CONVEX_CLOSED_BOUNDARY,
        straight_span_count=ClosedCount.FOUR,
    ) is EngineeringFeatureDisposition.NONMATCH

    assert evaluate_macro_convexity(
        negative_not_four, ConvexityKind.CONVEX_CLOSED_BOUNDARY
    ) is EngineeringFeatureDisposition.MATCH
    assert evaluate_macro_straight_span_count(
        negative_not_four, ClosedCount.FOUR
    ) is EngineeringFeatureDisposition.NONMATCH
    assert evaluate_positive_macro_conjunction(
        negative_not_four,
        convexity=ConvexityKind.CONVEX_CLOSED_BOUNDARY,
        straight_span_count=ClosedCount.FOUR,
    ) is EngineeringFeatureDisposition.NONMATCH


def test_micro_zigzags_and_repeated_stamps_never_change_macro_derivations() -> None:
    macro = _trace(SQUARE)
    zigzag = MicroTexturePrimitive(
        MicroTexturePrimitiveKind.ZIGZAG_STROKE,
        (_p(2, 2), _p(4, 3), _p(6, 1), _p(8, 3), _p(10, 1)),
    )
    stamps = tuple(
        MicroTexturePrimitive(kind, (_p(5 + index, 7),))
        for index, kind in enumerate(
            (
                MicroTexturePrimitiveKind.MARKER_CIRCLE,
                MicroTexturePrimitiveKind.MARKER_SQUARE,
                MicroTexturePrimitiveKind.MARKER_TRIANGLE,
            )
        )
    )
    plain = _evidence(macro)
    decorated = _evidence(macro, MicroTextureEvidence.complete((zigzag, *stamps)))

    assert plain.record_digest != decorated.record_digest
    assert plain.macro_action_trace.trace_digest == decorated.macro_action_trace.trace_digest
    assert derive_macro_convexity(plain) == derive_macro_convexity(decorated)
    assert derive_macro_straight_span_count(plain) == derive_macro_straight_span_count(
        decorated
    )
    assert decorated.to_data()["raw_black_ink_convex_hull_used"] is False
    assert decorated.micro_texture_evidence.to_data()["macro_geometry_effect"] == "none"


def test_concavity_comes_from_ordered_centerline_not_a_convex_hull() -> None:
    evidence = _evidence(_trace(CONCAVE_FOUR))
    derived = derive_macro_convexity(evidence)
    assert derived.status is GeometryDerivationStatus.RESOLVED
    assert derived.convexity_kind is ConvexityKind.CONCAVE_CLOSED_BOUNDARY
    assert derived.polygon is not None
    assert derived.to_data()["raw_black_ink_envelope_or_hull_consulted"] is False
    assert evidence.macro_action_trace.to_data()["macro_carrier"] == (
        "simplified_centerline_action_trace"
    )


@pytest.mark.parametrize(
    "resolution,issue",
    (
        (TraceResolution.INDETERMINATE, GeometryTraceIssue.MISSING_ORDERED_MACRO_TRACE),
        (TraceResolution.INDETERMINATE, GeometryTraceIssue.RESOLUTION_LIMIT),
        (TraceResolution.INDETERMINATE, GeometryTraceIssue.AMBIGUOUS_GEOMETRY),
    ),
)
def test_missing_or_failed_visual_fit_is_indeterminate_never_negative(
    resolution: TraceResolution, issue: GeometryTraceIssue
) -> None:
    evidence = _evidence(MacroActionTrace.gap(resolution, issue))
    assert evaluate_macro_convexity(
        evidence, ConvexityKind.CONVEX_CLOSED_BOUNDARY
    ) is EngineeringFeatureDisposition.INDETERMINATE
    assert evaluate_macro_straight_span_count(
        evidence, ClosedCount.FOUR
    ) is EngineeringFeatureDisposition.INDETERMINATE
    assert evaluate_positive_macro_conjunction(
        evidence,
        convexity=ConvexityKind.CONVEX_CLOSED_BOUNDARY,
        straight_span_count=ClosedCount.FOUR,
    ) is EngineeringFeatureDisposition.INDETERMINATE


@pytest.mark.parametrize(
    "issue",
    (
        GeometryTraceIssue.TRANSPORT_FAILURE,
        GeometryTraceIssue.PARSER_FAILURE,
        GeometryTraceIssue.INTEGRITY_FAILURE,
    ),
)
def test_protocol_failure_preserves_error_and_error_dominates_conjunction(
    issue: GeometryTraceIssue,
) -> None:
    evidence = _evidence(MacroActionTrace.gap(TraceResolution.ERROR, issue))
    assert derive_macro_convexity(evidence).status is GeometryDerivationStatus.ERROR
    assert (
        derive_macro_straight_span_count(evidence).status
        is GeometryDerivationStatus.ERROR
    )
    assert evaluate_macro_convexity(
        evidence, ConvexityKind.CONVEX_CLOSED_BOUNDARY
    ) is EngineeringFeatureDisposition.ERROR
    assert evaluate_macro_straight_span_count(
        evidence, ClosedCount.FOUR
    ) is EngineeringFeatureDisposition.ERROR
    assert evaluate_positive_macro_conjunction(
        evidence,
        convexity=ConvexityKind.CONVEX_CLOSED_BOUNDARY,
        straight_span_count=ClosedCount.FOUR,
    ) is EngineeringFeatureDisposition.ERROR


def test_interval_safe_arc_and_ambiguous_span_produce_a_count_range() -> None:
    a, b, c, d = SQUARE
    certain_arc = _arc(a, _p(8, 5), b)
    ambiguous_control = UncertainGrid16Point(
        Grid16Interval(12, 14), Grid16Interval(7, 9)
    )
    ambiguous_arc = _arc(b, ambiguous_control, c)
    trace = MacroActionTrace.complete(
        (certain_arc, ambiguous_arc, _line(c, d), _line(d, a))
    )
    evidence = _evidence(trace)
    derived = derive_macro_straight_span_count(evidence)

    assert derived.status is GeometryDerivationStatus.INDETERMINATE
    assert derived.span_kinds.count(DerivedMacroSpanKind.STRAIGHT) == 2
    assert derived.span_kinds.count(DerivedMacroSpanKind.CURVED) == 1
    assert derived.span_kinds.count(DerivedMacroSpanKind.INDETERMINATE) == 1
    assert (derived.lower_bound, derived.upper_bound) == (2, 3)
    assert evaluate_macro_straight_span_count(
        evidence, ClosedCount.FOUR
    ) is EngineeringFeatureDisposition.NONMATCH
    assert evaluate_macro_straight_span_count(
        evidence, ClosedCount.THREE
    ) is EngineeringFeatureDisposition.INDETERMINATE
    assert evaluate_macro_convexity(
        evidence, ConvexityKind.CONVEX_CLOSED_BOUNDARY
    ) is EngineeringFeatureDisposition.INDETERMINATE


def test_line_action_count_uses_typed_base_spans_not_rendered_trace_segments() -> None:
    evidence = _evidence(
        _trace(SQUARE),
        MicroTextureEvidence.complete(
            (
                MicroTexturePrimitive(
                    MicroTexturePrimitiveKind.ZIGZAG_STROKE,
                    tuple(_p(x, 2 + (x % 2)) for x in range(2, 14)),
                ),
            )
        ),
    )
    derived = derive_macro_straight_span_count(evidence)
    assert derived.status is GeometryDerivationStatus.RESOLVED
    assert (derived.lower_bound, derived.upper_bound) == (4, 4)
    assert len(evidence.micro_texture_evidence.primitives[0].points) == 12


def test_cycle_rotation_and_direction_have_one_canonical_identity() -> None:
    spans = tuple(
        _line(point, SQUARE[(index + 1) % len(SQUARE)])
        for index, point in enumerate(SQUARE)
    )
    rotated = spans[2:] + spans[:2]
    reversed_cycle = tuple(item.reversed() for item in reversed(spans))
    assert MacroActionTrace.complete(spans) == MacroActionTrace.complete(rotated)
    assert MacroActionTrace.complete(spans) == MacroActionTrace.complete(reversed_cycle)


def test_evidence_round_trip_tamper_rejection_and_model_free_cold_replay() -> None:
    evidence = _evidence(_trace(SQUARE))
    restored = HierarchicalActionGeometryEvidence.from_data(evidence.to_data())
    assert restored == evidence
    replay = HierarchicalActionGeometryReplay.create(restored)
    assert HierarchicalActionGeometryReplay.from_data(replay.to_data()) == replay
    assert cold_replay_hierarchical_action_geometry(
        replay, expected_replay_address=replay.replay_address
    ) == replay
    assert replay.to_data()["model_call_count"] == 0

    tampered = deepcopy(evidence.to_data())
    tampered["macro_action_trace"]["raw_ink_envelope_or_convex_hull_used"] = True
    with pytest.raises(HierarchicalActionGeometryError):
        HierarchicalActionGeometryEvidence.from_data(tampered)
    with pytest.raises(HierarchicalActionGeometryError):
        cold_replay_hierarchical_action_geometry(
            replay, expected_replay_address="sha256:" + "f" * 64
        )


def test_provenance_has_no_candidate_phase_class_formula_or_lean_channel() -> None:
    data = _provenance().to_data()
    assert data["candidate_specs_model_visible"] is False
    assert data["support_or_query_role_model_visible"] is False
    assert data["side_or_class_label_model_visible"] is False
    assert data["formula_model_visible"] is False
    assert data["lean_present"] is False
    assert GeometryEvidenceProvenance.from_data(data) == _provenance()
