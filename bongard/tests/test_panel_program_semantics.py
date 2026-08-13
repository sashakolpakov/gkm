from __future__ import annotations

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from bongard.canonical import canonical_digest
from bongard import panel_action_count_connected_synthetic as connected
from bongard.panel_program_observation import (
    PanelProgramObservation,
    PanelProgramObservationError,
    adapt_connected_fit_outcome,
    observe_connected_program_png,
)
from bongard.panel_program_predicate import (
    FrozenProgramRule,
    PanelProgramPredicateError,
    ProgramAtom,
    ProgramAxis,
    ProgramDisposition,
    ProgramFormula,
    ProgramSupportGap,
    ProgramSupportGapError,
    ProgramVersionSpace,
    build_program_version_space,
    cold_replay_frozen_program_rule,
    cold_replay_program_rule_decision,
    cold_replay_program_support_gap,
    cold_replay_program_version_space,
    enumerate_program_formulas,
    evaluate_frozen_program_rule,
    evaluate_program_formula,
    freeze_program_rule,
)


def _address(value: object) -> str:
    return "sha256:" + canonical_digest(value)


def _program_png(kind: str) -> bytes:
    primitive = next(row for row in connected.primitive_catalog() if row.kind == kind)
    return connected.render_catalog_program((primitive.primitive_id,))


@pytest.fixture(scope="module")
def observations() -> tuple[PanelProgramObservation, PanelProgramObservation]:
    return (
        observe_connected_program_png(_program_png("line")),
        observe_connected_program_png(_program_png("arc")),
    )


def _formula(*atoms: tuple[ProgramAxis, int | str]) -> ProgramFormula:
    return ProgramFormula.create(tuple(ProgramAtom.create(*item) for item in atoms))


def _same_png_count_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> PanelProgramObservation:
    """Build one real PNG with exact two-line and two-arc catalog covers."""

    from bongard import panel_action_count_connected_synthesizer as fitter

    pixels = tuple(32 * 64 + column for column in range(30, 34))
    rows = (
        ("toy-line-a", "line", (pixels[0], pixels[1])),
        ("toy-line-b", "line", (pixels[2], pixels[3])),
        ("toy-arc-a", "arc", (pixels[0], pixels[2])),
        ("toy-arc-b", "arc", (pixels[1], pixels[3])),
    )
    catalog = tuple(
        SimpleNamespace(
            primitive_id=primitive_id,
            kind=kind,
            ink_pixels=ink,
            endpoints_yx=tuple((pixel // 64, pixel % 64) for pixel in ink),
            boundary_pixels=ink,
        )
        for primitive_id, kind, ink in rows
    )
    image = np.full((64, 64), 255, dtype=np.uint8)
    image[32, 30:34] = 0
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=False)
    monkeypatch.setattr(connected, "require_issued_connected_png", lambda _raw: None)
    monkeypatch.setattr(connected, "primitive_catalog", lambda: catalog)
    fitter._catalog_masks.cache_clear()
    try:
        result = observe_connected_program_png(buffer.getvalue())
    finally:
        fitter._catalog_masks.cache_clear()
    assert result.state == "ambiguous"
    assert {(item.straight_count, item.arc_count) for item in result.hypotheses} == {
        (0, 2), (2, 0)
    }
    assert len({item.reconstructed_ink_pixels for item in result.hypotheses}) == 1
    return result


def _failed_observation(
    base: PanelProgramObservation, *, state: str
) -> PanelProgramObservation:
    content = {
        "schema": "gkm.panel-program-observation.v2",
        "panel_png_digest": _address({"synthetic": state}),
        "observer_source_digest": base.observer_source_digest,
        "observer_algorithm_digest": base.observer_algorithm_digest,
        "search_space_digest": base.search_space_digest,
        "hypothesis_policy_digest": base.hypothesis_policy_digest,
        "state": state,
        "reason": f"typed {state}",
        "error_type": "SyntheticObserverError" if state == "error" else None,
        "foreground_pixel_count": None,
        "skeleton_pixel_count": None,
        "minimum_primitive_count": None,
        "hypotheses": [],
    }
    return PanelProgramObservation(
        content["panel_png_digest"], content["observer_source_digest"],
        content["observer_algorithm_digest"], content["search_space_digest"],
        content["hypothesis_policy_digest"], state, content["reason"],
        content["error_type"], None, None, None, (), _address(content),
    )


def test_closed_inventory_is_exactly_32_singletons_plus_367_cross_axis_pairs() -> None:
    formulas = enumerate_program_formulas()
    assert len(formulas) == 399
    assert len({item.formula_digest for item in formulas}) == 399
    assert sum(len(item.atoms) == 1 for item in formulas) == 32
    assert sum(len(item.atoms) == 2 for item in formulas) == 367
    assert all(len({atom.axis for atom in item.atoms}) == len(item.atoms) for item in formulas)


def test_whole_formula_supervaluation_preserves_hypothesis_correlation(
    observations: tuple[PanelProgramObservation, PanelProgramObservation],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambiguous = _same_png_count_ambiguous(monkeypatch)
    straight = evaluate_program_formula(_formula((ProgramAxis.STRAIGHT, 2)), ambiguous)
    arc = evaluate_program_formula(_formula((ProgramAxis.ARC, 2)), ambiguous)
    conjunction = evaluate_program_formula(
        _formula((ProgramAxis.STRAIGHT, 2), (ProgramAxis.ARC, 2)), ambiguous
    )

    assert straight.disposition is ProgramDisposition.INDETERMINATE
    assert arc.disposition is ProgramDisposition.INDETERMINATE
    # Each concrete hypothesis falsifies the whole conjunction.  An atomwise
    # I-and-I composition would lose that correlation and get this wrong.
    assert conjunction.disposition is ProgramDisposition.CERTIFIED_ABSENT
    assert tuple(value for _, value in conjunction.hypothesis_truths) == (False, False)


def test_gap_and_error_never_become_boolean_false(
    observations: tuple[PanelProgramObservation, PanelProgramObservation],
) -> None:
    formula = _formula((ProgramAxis.TOTAL, 1))
    gap = evaluate_program_formula(formula, _failed_observation(observations[0], state="gap"))
    error = evaluate_program_formula(formula, _failed_observation(observations[0], state="error"))
    assert gap.disposition is ProgramDisposition.INDETERMINATE
    assert error.disposition is ProgramDisposition.ERROR
    assert not gap.hypothesis_truths and not error.hypothesis_truths


def test_strict_complete_version_space_freezes_and_cold_replays(
    observations: tuple[PanelProgramObservation, PanelProgramObservation],
) -> None:
    positive, contrast = observations
    space = build_program_version_space((positive,) * 6, (contrast,) * 6)
    assert space.candidate_count == 399
    assert len(space.matrix) == 399
    assert all(len(row) == 12 for row in space.matrix)
    assert space.survivor_count > 0
    assert ProgramVersionSpace.from_data(space.to_data()) == space
    assert cold_replay_program_version_space(
        space.to_data(), expected_digest=space.version_space_digest
    ) == space

    rule = freeze_program_rule(space)
    assert rule.formula_digest in space.survivor_formula_digests
    assert rule == FrozenProgramRule.from_data(rule.to_data())
    assert cold_replay_frozen_program_rule(
        rule.to_data(), expected_digest=rule.rule_digest
    ) == rule
    positive_decision = evaluate_frozen_program_rule(rule, positive)
    assert positive_decision.prediction == "positive"
    assert cold_replay_program_rule_decision(
        positive_decision.to_data(), rule=rule, observation=positive,
        expected_digest=positive_decision.decision_digest,
    ) == positive_decision
    assert evaluate_frozen_program_rule(rule, contrast).prediction == "contrast"


def test_zero_survivor_is_an_exact_semantic_gap(
    observations: tuple[PanelProgramObservation, PanelProgramObservation],
) -> None:
    line, _ = observations
    space = build_program_version_space((line,) * 6, (line,) * 6)
    assert space.survivor_count == 0
    assert space.survivor_formula_digests == ()
    with pytest.raises(ProgramSupportGapError, match="0/399 survivors") as captured:
        freeze_program_rule(space)
    gap = captured.value.gap
    assert type(gap) is ProgramSupportGap
    assert gap.version_space_digest == space.version_space_digest
    assert gap.gap_kind == "language_gap"
    assert gap.error_cell_count == gap.indeterminate_cell_count == 0
    assert cold_replay_program_support_gap(
        gap.to_data(), version_space=space, expected_digest=gap.gap_digest
    ) == gap


def test_support_failure_distinguishes_observer_error_from_language_gap(
    observations: tuple[PanelProgramObservation, PanelProgramObservation],
) -> None:
    line, arc = observations
    error = _failed_observation(line, state="error")
    space = build_program_version_space((error,) * 6, (arc,) * 6)
    with pytest.raises(ProgramSupportGapError) as captured:
        freeze_program_rule(space)
    gap = captured.value.gap
    assert gap.gap_kind == "observation_error"
    assert gap.error_cell_count == 399 * 6
    assert gap.error_observation_digests == (error.observation_digest,)


def test_adaptation_is_target_free_and_canonical_mutation_fails(
    observations: tuple[PanelProgramObservation, PanelProgramObservation],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png = _program_png("line")
    from bongard import panel_action_count_connected_synthesizer as fitter

    outcome = fitter.fit_png_hypotheses(png)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("semantic adaptation consulted target/program history")

    monkeypatch.setattr(connected, "exact_cover_target", forbidden)
    adapted = adapt_connected_fit_outcome(png, outcome)
    assert adapted == observations[0]
    with pytest.raises(PanelProgramObservationError, match="canonical replay"):
        adapt_connected_fit_outcome(_program_png("arc"), outcome)

    forged = deepcopy(adapted.to_data())
    forged["hypotheses"][0]["straight_count"] = 9
    with pytest.raises((PanelProgramObservationError, ValueError, TypeError)):
        PanelProgramObservation.from_data(forged)
