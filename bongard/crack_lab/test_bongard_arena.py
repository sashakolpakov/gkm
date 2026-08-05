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
import predicate_pricing as predicate_price

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


@pytest.mark.parametrize("lower,upper", [
    (1.0, np.nextafter(1.0, np.inf)),
    (1.0e308, 1.7e308),
])
def test_selector_keeps_both_orientations_at_extreme_float_boundaries(
        lower, upper):
    values = np.array([[upper], [upper], [lower], [lower]], dtype=float)
    labels = np.array([True, True, False, False])
    rule = B.select_rule(values, ["p_boundary"], labels)
    assert rule.constant is None
    assert len(rule.atoms) == 1
    assert rule.atoms[0].name == "p_boundary"
    assert rule.atoms[0].op == ">="
    assert float(lower) < rule.atoms[0].threshold <= float(upper)
    assert [rule.predict({"p_boundary": value}) for value in values[:, 0]] \
        == labels.tolist()


def test_priced_selector_uses_exact_transitive_definition_cost():
    source = """\
TABLE = [1, 2, 3, 4, 5, 6, 7, 8]

def p_a(panel):
    return float(panel.sum()) + 0.0 * sum(TABLE)

def p_z(panel):
    return float(panel.sum())
"""
    problem = two_vs_one_problem()
    no_share = B.verify_priced_source(
        source, problem, sharing_policy=B.NO_SHARE_PRICING)
    assert no_share.solved
    assert no_share.predicate_names == ("p_z",)
    assert no_share.definition_cost == no_share.full_definition_cost

    model = predicate_price.build_pricing_model(source)
    paid_a = model.identities_for(["p_a"])
    shared = B.verify_priced_source(
        source, problem, sharing_policy=B.SHARED_PRICING,
        paid_node_identities=paid_a)
    assert shared.solved
    assert shared.predicate_names == ("p_a",)
    assert shared.definition_cost == 0
    assert set(shared.used_definition_node_identities) == paid_a


def test_definition_union_is_paid_once_but_bindings_remain_per_atom():
    source = "def p_a(panel):\n    return float(panel.sum())\n"
    model = predicate_price.build_pricing_model(source)
    pricing = B.RulePricing(model, B.NO_SHARE_PRICING)
    rule = B.Rule(atoms=(
        B.Atom("p_a", ">=", 1.0),
        B.Atom("p_a", "<=", 1000.0),
    ))
    receipt = model.price_no_share(["p_a"])
    assert rule.predicate_names == ("p_a",)
    assert rule.structure_cost() == 2 * (B.CALL_COST + B.BINDING_COST)
    assert rule.cost(pricing) == rule.structure_cost() + receipt.full_cost


def test_priced_conditional_mdl_does_not_make_constant_unbeatable():
    table = ", ".join(str(i) for i in range(80))
    source = (
        f"TABLE = [{table}]\n"
        "def p_expensive(panel):\n"
        "    return float(panel.sum()) + 0.0 * sum(TABLE)\n"
    )
    result = B.verify_priced_source(
        source, two_vs_one_problem(), sharing_policy=B.NO_SHARE_PRICING)
    assert result.solved
    assert result.definition_cost > 5
    assert result.selection_policy == B.PRICED_SELECTION_POLICY


def test_priced_source_rejects_runtime_ast_predicate_mismatch():
    source = """\
from math import sin as p_imported

def p_ink(panel):
    return float(panel.sum())
"""
    with pytest.raises(ValueError, match="runtime p_\\* callables"):
        B.verify_priced_source(
            source, two_vs_one_problem(), sharing_policy=B.SHARED_PRICING)


def test_priced_source_rejects_evaluation_order_classifier():
    source = """\
COUNTER = [0]

def p_order(panel):
    COUNTER[0] += 1
    return COUNTER[0] <= 6
"""
    with pytest.raises(predicate_price.PredicatePricingError,
                       match="aliased state|locally owned"):
        B.verify_priced_source(
            source, two_vs_one_problem(), sharing_policy=B.SHARED_PRICING)


def test_priced_source_rejects_crashes_and_nonfinite_values():
    crashing = """\
def p_bad(panel):
    raise RuntimeError('boom')

def p_ink(panel):
    return float(panel.sum())
"""
    with pytest.raises(B.PredicateEvaluationError, match="failed on panel"):
        B.verify_priced_source(
            crashing, two_vs_one_problem(), sharing_policy=B.SHARED_PRICING)

    nonfinite = "def p_bad(panel):\n    return float('nan')\n"
    with pytest.raises(B.PredicateEvaluationError, match="non-finite"):
        B.verify_priced_source(
            nonfinite, two_vs_one_problem(), sharing_policy=B.SHARED_PRICING)


def test_priced_source_rejects_even_stable_module_cache_mutation():
    source = """\
CACHE = {}

def p_ink(panel):
    CACHE[0] = float(panel.sum())
    return CACHE[0]
"""
    with pytest.raises(predicate_price.PredicatePricingError,
                       match="locally owned"):
        B.verify_priced_source(
            source, two_vs_one_problem(), sharing_policy=B.SHARED_PRICING)


def test_pure_local_scratch_predicate_is_stable_across_audit_orders():
    source = """\
import numpy as np

def p_ink(panel):
    values = []
    values.append(float(panel.sum()))
    scratch = np.zeros(2)
    scratch[0] = values[0]
    scratch[1] = 0.0
    return float(scratch.sum())
"""
    result = B.verify_priced_source(
        source, two_vs_one_problem(), sharing_policy=B.SHARED_PRICING)
    assert result.solved


def test_predicate_evaluation_preserves_caller_panels_even_in_legacy_mode():
    panel = np.arange(16, dtype=float).reshape(4, 4)
    original = panel.copy()

    def p_mutating(received):
        received[:, :] = -1
        return float(received.sum())

    values, names, errors = B.predicate_values(
        {"p_mutating": p_mutating}, [panel], strict=False)
    assert names == ["p_mutating"]
    assert errors == 0
    assert values.shape == (1, 1)
    assert np.array_equal(panel, original)


def test_public_loader_applies_pricing_gate_and_restricted_builtins():
    with pytest.raises(predicate_price.PredicatePricingError):
        B.load_predicates_source(
            "def p_bad(panel):\n    return help(panel)\n"
        )
    predicates = B.load_predicates_source(
        "def p_ok(panel):\n    return float(panel.sum())\n"
    )
    builtins_namespace = predicates["p_ok"].__globals__["__builtins__"]
    assert "float" in builtins_namespace
    assert "help" not in builtins_namespace
    assert "open" not in builtins_namespace


def test_file_loader_uses_bounded_predicate_reader(tmp_path):
    oversized = tmp_path / "predicates.py"
    oversized.write_bytes(
        b"x" * (predicate_price.MAX_SOURCE_UTF8_BYTES + 1))
    with pytest.raises(predicate_price.PredicatePricingError,
                       match="UTF-8 byte limit"):
        B.load_predicates(str(oversized))


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


def test_write_panels_rejects_workspace_and_problem_symlinks_without_mutation(
        tmp_path):
    problem = two_vs_one_problem()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("untouched")

    linked_workspace = tmp_path / "linked_workspace"
    linked_workspace.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="non-symlink directory"):
        B.write_panels(str(linked_workspace), problem, "problem_00")
    assert sentinel.read_text() == "untouched"
    assert sorted(path.name for path in outside.iterdir()) == ["sentinel.txt"]

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    linked_problem = workspace / "problem_00"
    linked_problem.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="failed to materialize"):
        B.write_panels(str(workspace), problem, "problem_00")
    assert sentinel.read_text() == "untouched"
    assert linked_problem.is_symlink()


def test_write_panels_rejects_non_directory_workspace_and_problem(
        tmp_path):
    problem = two_vs_one_problem()
    workspace_file = tmp_path / "workspace_file"
    workspace_file.write_text("workspace-sentinel")
    with pytest.raises(RuntimeError, match="non-symlink directory"):
        B.write_panels(str(workspace_file), problem, "problem_00")
    assert workspace_file.read_text() == "workspace-sentinel"

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    problem_file = workspace / "problem_00"
    problem_file.write_text("problem-sentinel")
    with pytest.raises(RuntimeError, match="failed to materialize"):
        B.write_panels(str(workspace), problem, "problem_00")
    assert problem_file.read_text() == "problem-sentinel"


def test_write_panels_rejects_panel_symlink_and_extra_file_without_mutation(
        tmp_path):
    problem = two_vs_one_problem()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    panel_dir = workspace / "problem_00"
    B.write_panels(str(workspace), problem, "problem_00")

    sentinel = tmp_path / "outside.npy"
    sentinel.write_bytes(b"outside-sentinel")
    panel_link = panel_dir / "pos_0.npy"
    panel_link.unlink()
    panel_link.symlink_to(sentinel)
    before = {
        path.name: ("link", os.readlink(path)) if path.is_symlink()
        else ("file", path.read_bytes())
        for path in panel_dir.iterdir()
    }
    with pytest.raises(RuntimeError, match="failed to materialize"):
        B.write_panels(str(workspace), problem, "problem_00")
    after = {
        path.name: ("link", os.readlink(path)) if path.is_symlink()
        else ("file", path.read_bytes())
        for path in panel_dir.iterdir()
    }
    assert after == before
    assert sentinel.read_bytes() == b"outside-sentinel"

    panel_link.unlink()
    panel_link.write_bytes(b"owned-placeholder")
    extra = panel_dir / "unowned.txt"
    extra.write_text("keep me")
    before = {path.name: path.read_bytes() for path in panel_dir.iterdir()}
    with pytest.raises(RuntimeError, match="failed to materialize"):
        B.write_panels(str(workspace), problem, "problem_00")
    assert {path.name: path.read_bytes() for path in panel_dir.iterdir()} == before


def test_write_panels_rejects_panel_hardlink_without_mutating_target(tmp_path):
    problem = two_vs_one_problem()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    panel_dir = workspace / "problem_00"
    B.write_panels(str(workspace), problem, "problem_00")
    outside = tmp_path / "outside.npy"
    outside.write_bytes(b"outside-hardlink-sentinel")
    panel_path = panel_dir / "pos_0.npy"
    panel_path.unlink()
    os.link(outside, panel_path)
    before = {
        path.name: path.read_bytes()
        for path in panel_dir.iterdir()
    }
    with pytest.raises(RuntimeError, match="failed to materialize"):
        B.write_panels(str(workspace), problem, "problem_00")
    assert outside.read_bytes() == b"outside-hardlink-sentinel"
    assert {path.name: path.read_bytes() for path in panel_dir.iterdir()} == before


@pytest.mark.parametrize("opaque_id", ("../escape", "/absolute", "problem_1"))
def test_write_panels_rejects_noncanonical_opaque_id_before_mutation(
        tmp_path, opaque_id):
    before = sorted(path.name for path in tmp_path.iterdir())
    with pytest.raises(ValueError, match="canonical form"):
        B.write_panels(str(tmp_path), two_vs_one_problem(), opaque_id)
    assert sorted(path.name for path in tmp_path.iterdir()) == before
