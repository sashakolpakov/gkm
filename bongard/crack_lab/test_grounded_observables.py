"""Integration contracts for candidate-independent grounded observables."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import grounded_predicate_ir as G
import grounded_observables as O
import semantic_legs as L
from dataset import sample_problem_programs


DATASET = Path(__file__).parents[2] / "downloads" / "Bongard-LOGO"


def _truths(compiled: G.CompiledPredicate, panels) -> list[bool]:
    values: list[bool] = []
    for panel in panels:
        result = compiled.evaluate(O.GroundedPanelContext(panel))
        assert isinstance(result, G.Present), result
        assert result.unit is G.Unit.BOOLEAN
        values.append(bool(result.value))
    return values


def test_all_formula_leaves_share_one_candidate_independent_extraction(
        monkeypatch) -> None:
    calls = 0

    def absent(_panel):
        nonlocal calls
        calls += 1
        raise L.WitnessAbsent(
            "no_point_contact_signature", "certified fixture absence")

    monkeypatch.setattr(L, "extract_point_contact_signature", absent)
    registry, _descriptors = O.default_grounded_observables()
    predicate = G.All((
        G.Compare(O.SMALL_GAP_ID, "lt", G.Literal(45.0, "degrees")),
        G.Compare(O.LARGE_GAP_ID, "gt", G.Literal(90.0, "degrees")),
        G.Compare(O.GAP_RATIO_ID, "gt", G.Literal(2.5, "ratio")),
    ))
    trace = G.compile_predicate(predicate, registry).evaluate_with_trace(
        O.GroundedPanelContext(np.zeros((32, 32), dtype=np.uint8)))

    assert trace.result == G.Present(False, G.Unit.BOOLEAN)
    assert calls == 1
    assert len(trace.observations) == 3
    assert all(isinstance(value, G.SemanticAbsent)
               for _observable_id, value in trace.observations)


def test_failed_fit_is_unknown_and_never_negative_evidence(monkeypatch) -> None:
    def unresolved(_panel):
        raise L.WitnessIndeterminate(
            "point_contact_fit_indeterminate", "rays cannot be fitted")

    monkeypatch.setattr(L, "extract_point_contact_signature", unresolved)
    registry, _descriptors = O.default_grounded_observables()
    compiled = G.compile_predicate(
        G.Compare(O.SMALL_GAP_ID, "lt", G.Literal(45.0, "degrees")),
        registry,
    )
    result = compiled.evaluate(
        O.GroundedPanelContext(np.zeros((32, 32), dtype=np.uint8)))
    assert isinstance(result, G.Indeterminate)
    assert result.mode == "point_contact_fit_indeterminate"


@pytest.mark.skipif(
    not (DATASET / "data" / "human_designed_shapes.tsv").is_file(),
    reason="Bongard-LOGO latent programs unavailable",
)
def test_frozen_grounded_atoms_generalize_to_unseen_rerenders() -> None:
    latent = sample_problem_programs(
        str(DATASET), limit=1, seed=20260805, source="basic")[0]
    assert latent.concept == "mismatch_sector_rec2"
    registry, _descriptors = O.default_grounded_observables()
    predicates = (
        G.Compare(O.SMALL_GAP_ID, "lt", G.Literal(45.0, "degrees")),
        G.Compare(O.GAP_RATIO_ID, "gt", G.Literal(2.5, "ratio")),
    )

    for predicate in predicates:
        compiled = G.compile_predicate(predicate, registry)
        assert compiled.taint is G.Taint.PURE
        for render_seed in (20260805, 20260806, 20260807, 20260905):
            problem = latent.render(render_seed)
            assert _truths(compiled, problem.pos + problem.neg) == \
                [True] * 6 + [False] * 6
