"""Offline tests for the enforced predicate-library orchestration.

The proposer is injected; no LLM, no dataset, no network. Witness predicate
code lives only in these tests (representability floor, never shipped)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bongard_arena as B
import bongard_legs as L
from test_bongard_arena import two_vs_one_problem, SQUARE, CIRCLE, _make_problem

WITNESS = "def p_ink(panel):\n    return float(panel.sum())\n"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect the artifact root into the sandbox so tests never touch the
    repo's agent_solutions."""
    monkeypatch.setattr(L, "artifact_dir",
                        lambda tag: str(tmp_path / f"art_{tag}"))
    return tmp_path


def _two_problems(seed=5):
    return [two_vs_one_problem(),
            _make_problem([[SQUARE, CIRCLE]] * 6,
                          [[CIRCLE]] * 3 + [[SQUARE]] * 3, seed=seed)]


def writing_proposer(code=WITNESS):
    def propose(task, ws, model, minutes):
        with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
            f.write(code)
    return propose


def test_solved_problems_and_reuse_is_free(sandbox):
    """First problem pays for the witness predicate; the second reuses it
    for marginal_C == 0 (the sawtooth's reuse floor)."""
    calls = []

    def propose(task, ws, model, minutes):
        calls.append(model)
        lib = os.path.join(ws, L.LIBRARY_FILE)
        if "p_ink" not in open(lib).read():
            with open(lib, "a") as f:
                f.write(WITNESS)

    rep = L.run(_two_problems(), tag="t1", ws=str(sandbox / "ws1"),
                propose_fn=propose, verbose=False)
    assert rep.solved == 2
    assert rep.records[0].marginal_C > 0
    assert rep.records[1].marginal_C == 0
    assert calls == ["sonnet", "sonnet"]
    assert not any(r.escalated for r in rep.records)


def test_failed_attempt_reverts_library_and_saves_wip(sandbox):
    """Structural admission: junk written during an unsolved problem must not
    enter the shared library, but survives as WIP context."""
    def propose(task, ws, model, minutes):
        with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
            f.write("def p_useless(panel):\n    return 0.0\n")

    ws = str(sandbox / "ws2")
    rep = L.run([two_vs_one_problem()], tag="t2", ws=ws,
                propose_fn=propose, verbose=False)
    assert rep.solved == 0
    assert rep.records[0].attempts == len(L.DEFAULT_LADDER)
    assert rep.records[0].escalated
    assert "p_useless" not in open(os.path.join(ws, L.LIBRARY_FILE)).read()
    wip = os.path.join(L.artifact_dir("t2"), "wip_context", "problem_00")
    assert os.path.isdir(wip) and os.listdir(wip)


def test_escalation_ladder_logged(sandbox):
    """Sonnet fails, Opus succeeds -> escalated=True, model=opus."""
    def propose(task, ws, model, minutes):
        if model == "opus":
            with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
                f.write(WITNESS)

    rep = L.run([two_vs_one_problem()], tag="t3", ws=str(sandbox / "ws3"),
                propose_fn=propose, ladder=("sonnet", "opus"), verbose=False)
    assert rep.solved == 1
    assert rep.records[0].model == "opus"
    assert rep.records[0].escalated


def test_taint_refuses_promotion(sandbox):
    def propose(task, ws, model, minutes):
        with open(os.path.join(ws, "notes.md"), "w") as f:
            f.write("peeked at human_designed_shapes.tsv for the answer")
        with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
            f.write(WITNESS)

    with pytest.raises(L.WorkspaceTainted):
        L.run([two_vs_one_problem()], tag="t4", ws=str(sandbox / "ws4"),
              propose_fn=propose, verbose=False)


def test_resume_skips_solved_and_ground_truth_stays_out_of_ws(sandbox):
    ws = str(sandbox / "ws5")
    rep1 = L.run(_two_problems(), tag="t5", ws=ws,
                 propose_fn=writing_proposer(), verbose=False)
    assert rep1.solved == 2
    # workspace must not contain concept names or results.json
    for root, _dirs, files in os.walk(ws):
        assert "results.json" not in files
        for name in files:
            if name.endswith((".py", ".md", ".json", ".txt")):
                text = open(os.path.join(root, name)).read()
                assert "two_shapes_vs_one" not in text
    art = L.artifact_dir("t5")
    results = json.load(open(os.path.join(art, "results.json")))
    assert results["problem_00"]["concept"] == "two_shapes_vs_one"

    def must_not_be_called(task, ws_, model, minutes):
        raise AssertionError("solved problems must not re-run the proposer")

    rep2 = L.run(_two_problems(), tag="t5", ws=str(sandbox / "ws5b"),
                 propose_fn=must_not_be_called, verbose=False)
    assert rep2.solved == 2


def test_literal_cost_charges_lookup_tables():
    honest = "def p_a(panel):\n    return float(panel.sum())\n"
    table = "def p_a(panel):\n    return T[hash(panel.tobytes()) % 12]\nT = [" \
            + ", ".join(["1.0"] * 12) + "]\n"
    assert L.description_complexity(table) > L.description_complexity(honest)
