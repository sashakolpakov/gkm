"""Offline tests for the raw Bongard substrate.

Witness predicates live ONLY here: they are representability floors for
testing the harness machinery, never shipped to the proposer (the no-hand-
coding rule). Sampler-dependent tests skip when downloads/Bongard-LOGO is
absent.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bongard_arena as B

DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "downloads", "Bongard-LOGO")

SQUARE = ["line_normal_0.500-0.500", "line_normal_0.500-0.750",
          "line_normal_0.500-0.750", "line_normal_0.500-0.750"]
CIRCLE = ["arc_normal_0.300_1.000-0.500"]


def _witness_ink(panel):
    return float(panel.sum())


def _make_problem(pos_programs, neg_programs, seed=7):
    rng = np.random.RandomState(seed)
    pos = [B.render_panel(p, np.random.RandomState(rng.randint(2**31)))
           for p in pos_programs]
    neg = [B.render_panel(p, np.random.RandomState(rng.randint(2**31)))
           for p in neg_programs]
    return B.Problem("test_problem", "test", "two_shapes_vs_one", pos, neg)


def two_vs_one_problem():
    """Positives contain two shapes, negatives one -- separable by ink mass."""
    return _make_problem([[SQUARE, CIRCLE]] * 6, [[SQUARE]] * 3 + [[CIRCLE]] * 3)


def test_trace_square_closes():
    pts = B.trace_shape(SQUARE)
    assert len(pts) == 5
    x, y = pts[-1]
    assert abs(x) < 1e-9 and abs(y) < 1e-9


def test_render_deterministic_and_inked():
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    a = B.render_panel([SQUARE], rng1)
    b = B.render_panel([SQUARE], rng2)
    assert np.array_equal(a, b)
    assert a.sum() > 50
    assert a.shape == (B.PANEL_SIZE, B.PANEL_SIZE)
    assert set(np.unique(a)) <= {0, 1}


def test_verify_solves_with_witness_predicate():
    problem = two_vs_one_problem()
    res = B.verify({"p_ink": _witness_ink}, problem)
    assert res.solved
    assert res.heldout_accuracy == 1.0
    assert "p_ink" in res.rule
    assert res.rule_cost == B.CALL_COST + B.BINDING_COST


def test_verify_replay_is_bit_exact():
    problem = two_vs_one_problem()
    r1 = B.verify({"p_ink": _witness_ink}, problem)
    r2 = B.verify({"p_ink": _witness_ink}, problem)
    assert r1 == r2


def test_verify_fails_without_predicates():
    problem = two_vs_one_problem()
    res = B.verify({}, problem)
    assert not res.solved


def test_shuffled_sides_control_fails():
    """The structural control: reassign panels to sides at random -> the same
    witness predicate must NOT produce a solved verdict."""
    problem = two_vs_one_problem()
    rng = np.random.RandomState(0)
    panels = problem.pos + problem.neg
    order = rng.permutation(12)
    shuffled = B.Problem("shuffled", "control", "shuffled",
                         [panels[i] for i in order[:6]],
                         [panels[i] for i in order[6:]])
    res = B.verify({"p_ink": _witness_ink}, shuffled)
    assert not res.solved


def test_crashing_predicate_is_counted_not_fatal():
    def p_bad(panel):
        raise RuntimeError("boom")
    problem = two_vs_one_problem()
    res = B.verify({"p_bad": p_bad, "p_ink": _witness_ink}, problem)
    assert res.solved
    assert res.predicate_errors == 12


def test_select_rule_prefers_cheaper_on_ties():
    values = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    labels = np.array([True, True, False, False])
    rule = B.select_rule(values, ["p_a", "p_b"], labels)
    assert len(rule.atoms) == 1


@pytest.mark.skipif(not os.path.isdir(DATASET),
                    reason="downloads/Bongard-LOGO not present")
def test_sample_problems_deterministic():
    ps1 = B.sample_problems(DATASET, limit=2, seed=11, source="basic")
    ps2 = B.sample_problems(DATASET, limit=2, seed=11, source="basic")
    assert len(ps1) == 2
    for a, b in zip(ps1, ps2):
        assert a.problem_id == b.problem_id
        assert all(np.array_equal(x, y) for x, y in zip(a.pos, b.pos))
        assert all(np.array_equal(x, y) for x, y in zip(a.neg, b.neg))
        assert len(a.pos) == 6 and len(a.neg) == 6
        assert all(p.sum() > 20 for p in a.pos + a.neg)


@pytest.mark.skipif(not os.path.isdir(DATASET),
                    reason="downloads/Bongard-LOGO not present")
def test_write_panels_hides_concept(tmp_path):
    ps = B.sample_problems(DATASET, limit=1, seed=3, source="basic")
    pdir = B.write_panels(str(tmp_path), ps[0], "problem_00")
    names = os.listdir(pdir)
    assert any(n.endswith(".npy") for n in names)
    joined = " ".join(names) + " " + pdir
    assert ps[0].concept not in joined
    assert ps[0].problem_id not in joined
