"""Regression barriers for the Phase-D v3 foreground/sentinel failure.

The legacy unrestricted verifier deliberately has a finite-scalar-only
predicate API.  It must reject explicit invalid values before selection; a
finite number such as ``-1`` is *not* an invalidity marker and is therefore a
dangerous substitute.  Typed structural absence belongs to the semantic path,
where it is represented as ``None`` and excluded from threshold fitting.

The authoritative array and presentation encodings intentionally have
opposite visual polarity: predicate arrays use ink=1/background=0, while PNGs
render ink as black and background as white.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bongard_arena as raw
from dataset import Problem
from semantic_compiler import compile_hypothesis
from semantic_ir import (
    DiagramEdge,
    DiagramSpec,
    LegCall,
    MorphSpec,
    SemanticHypothesis,
)
from semantic_legs import default_registry
from semantic_verifier import _fit_threshold, verify_hypothesis


def _binary_problem(*, positive_ink: int = 100, negative_ink: int = 25) \
        -> raw.Problem:
    def panel(area: int, offset: int) -> np.ndarray:
        height = 10 if area == 100 else 5
        width = area // height
        result = np.zeros((raw.PANEL_SIZE, raw.PANEL_SIZE), dtype=np.uint8)
        result[30 + offset:30 + offset + height, 40:40 + width] = 1
        return result

    return raw.Problem(
        "v3_contract_fixture",
        "fixture",
        "harness_only",
        [panel(positive_ink, offset) for offset in range(6)],
        [panel(negative_ink, offset) for offset in range(6)],
    )


@pytest.mark.parametrize(
    ("source", "error_pattern"),
    (
        ("def p_invalid(panel):\n    return None\n", "failed on panel"),
        ("def p_invalid(panel):\n    return float('nan')\n", "non-finite"),
    ),
)
def test_authoritative_raw_invalid_measurement_fails_before_threshold_selection(
        monkeypatch, source, error_pattern):
    """Neither an explicit missing value nor NaN can become a rule atom."""
    def selection_must_not_run(*_args, **_kwargs):
        raise AssertionError("threshold selection saw an invalid measurement")

    monkeypatch.setattr(raw, "select_rule", selection_must_not_run)
    with pytest.raises(raw.PredicateEvaluationError, match=error_pattern):
        raw.verify_priced_source(
            source,
            _binary_problem(),
            sharing_policy=raw.SHARED_PRICING,
        )


def test_finite_sentinel_is_numeric_and_can_contaminate_legacy_selection():
    """Reproduce the terminal v3 score geometry as a diagnostic, not an API.

    Eleven panels returned ``-1`` and only the final negative returned a real
    turning score.  Because ``-1`` is finite, the legacy selector necessarily
    treats it as evidence and obtains the observed 7/12 training split.  New
    semantic implementations must use typed absence instead of copying this
    compatibility behavior.
    """
    values = np.asarray(
        [[-1.0]] * 11 + [[14.213492027047863]], dtype=float)
    labels = np.asarray([True] * 6 + [False] * 6, dtype=bool)

    rule = raw.select_rule(
        values,
        ["p_endpoint_turn_degrees"],
        labels,
        lam=0.0,
        max_atoms=1,
    )

    assert rule.constant is None
    assert len(rule.atoms) == 1
    atom = rule.atoms[0]
    assert atom.op == "<="
    assert atom.threshold == pytest.approx(6.6067460135239315)
    predictions = np.asarray([
        rule.predict({"p_endpoint_turn_degrees": row[0]})
        for row in values
    ])
    assert int(np.sum(predictions == labels)) == 7
    assert predictions[6:11].tolist() == [True] * 5


def test_panel_array_and_png_polarities_are_explicit_and_inverse(tmp_path):
    """Arrays are ink=1/background=0; viewable PNGs are black-on-white."""
    panel = np.zeros((raw.PANEL_SIZE, raw.PANEL_SIZE), dtype=np.uint8)
    panel[32:40, 44:52] = 1
    problem = raw.Problem(
        "hidden", "fixture", "harness_only",
        [panel.copy() for _ in range(6)],
        [panel.copy() for _ in range(6)],
    )
    panel_dir = raw.write_panels(str(tmp_path), problem, "problem_00")

    array = np.load(os.path.join(panel_dir, "pos_0.npy"), allow_pickle=False)
    with Image.open(os.path.join(panel_dir, "pos_0.png")) as encoded:
        presentation = np.asarray(encoded.convert("L"))

    assert array.dtype == np.uint8
    assert set(np.unique(array)) == {0, 1}
    assert array[35, 47] == 1       # authoritative predicate input: ink
    assert array[0, 0] == 0         # authoritative predicate input: background
    assert presentation[35, 47] == 0    # presentation: black ink
    assert presentation[0, 0] == 255    # presentation: white background
    assert np.array_equal(
        array, (presentation == 0).astype(np.uint8))


def _more_ink_hypothesis(*, order: str = "high_positive",
                         polarity: str = "positive_satisfies") \
        -> SemanticHypothesis:
    return SemanticHypothesis(
        version="0.1",
        hypothesis_id="more_total_ink",
        description="Positive panels have more total ink.",
        polarity=polarity,
        diagram=DiagramSpec((
            DiagramEdge("score", LegCall("total_ink", ("panel",))),
        )),
        score_node="score",
        order=order,
        preservation_morphisms=(MorphSpec("translate", "panel"),),
        semantic_requirements=("more total ink",),
    )


def _component_count_problem() -> Problem:
    def panel(two: bool, offset: int) -> np.ndarray:
        result = np.zeros(
            (raw.PANEL_SIZE, raw.PANEL_SIZE), dtype=np.uint8)
        result[24 + offset:36 + offset, 24:36] = 1
        if two:
            result[78 - offset:90 - offset, 86:98] = 1
        return result

    return Problem(
        "component_polarity_fixture",
        "fixture",
        "harness_only",
        tuple(panel(True, offset) for offset in range(6)),
        tuple(panel(False, offset) for offset in range(6)),
    )


def test_declared_semantic_score_direction_conflict_cannot_be_selected():
    """Prose saying 'more' cannot be executed with low-positive polarity."""
    hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id="more_components_wrong_direction",
        description="Positive panels have a higher object count.",
        polarity="positive_satisfies",
        diagram=DiagramSpec((
            DiagramEdge("scene", LegCall("parse_scene", ("panel",))),
            DiagramEdge("score", LegCall("object_count", ("scene",))),
        )),
        score_node="score",
        order="low_positive",
        preservation_morphisms=(MorphSpec("translate", "panel"),),
        semantic_requirements=("higher object count",),
    )
    result = verify_hypothesis(
        hypothesis,
        default_registry(),
        _component_count_problem(),
    )

    assert not result.accepted
    assert result.semantic_issue == (
        "semantic_score_direction_mismatch:high:low_positive")


def test_unsupported_semantic_header_polarity_fails_at_compile_time():
    with pytest.raises(ValueError, match="only positive_satisfies polarity"):
        compile_hypothesis(
            _more_ink_hypothesis(polarity="negative_satisfies"),
            default_registry(),
        )


def test_typed_structural_absence_is_none_and_never_a_threshold_value():
    """WitnessAbsent follows the semantic absence channel, not a sentinel."""
    empty = np.zeros((raw.PANEL_SIZE, raw.PANEL_SIZE), dtype=np.uint8)
    negatives = []
    for offset in range(6):
        panel = np.zeros_like(empty)
        panel[30 + offset:50 + offset, 42:82] = 1
        negatives.append(panel)
    problem = Problem(
        "typed_absence_fixture",
        "fixture",
        "harness_only",
        tuple(empty.copy() for _ in range(6)),
        tuple(negatives),
    )
    hypothesis = SemanticHypothesis(
        version="0.1",
        hypothesis_id="absent_principal_object",
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

    assert result.predicate_errors == 0
    assert result.structural_absences == 6
    assert result.scores[:6] == (None,) * 6
    assert all(score is not None for score in result.scores[6:])
    assert result.support_predictions[:6] == (False,) * 6
    assert result.support_errors >= 6
    assert not result.accepted

    numeric_scores = np.asarray([
        np.nan if score is None else score for score in result.scores
    ], dtype=float)
    labels = np.asarray(result.support_labels, dtype=bool)
    finite = np.isfinite(numeric_scores)
    finite_only_rule = _fit_threshold(
        numeric_scores[finite], labels[finite], hypothesis.order)
    all_rows_rule = _fit_threshold(numeric_scores, labels, hypothesis.order)
    assert all_rows_rule.threshold == pytest.approx(
        finite_only_rule.threshold)
