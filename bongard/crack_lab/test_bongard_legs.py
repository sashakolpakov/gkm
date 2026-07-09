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


def test_interrupted_workspace_is_snapshotted_before_seed(sandbox):
    """Power-out fallback: an in-flight library edit that differs from the
    promoted artifact must be preserved as WIP, not silently overwritten."""
    ws = str(sandbox / "ws6")
    rep1 = L.run([two_vs_one_problem()], tag="t6", ws=ws,
                 propose_fn=writing_proposer(), verbose=False)
    assert rep1.solved == 1
    # simulate an interrupted next attempt: live edits + current problem marker
    with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
        f.write("def p_inflight(panel):\n    return 1.0\n")
    with open(os.path.join(ws, "current_problem.txt"), "w") as f:
        f.write("problem_01")
    L.seed_workspace_from_artifact("t6", ws, verbose=False)
    # workspace was restored to the verified artifact...
    assert "p_inflight" not in open(os.path.join(ws, L.LIBRARY_FILE)).read()
    # ...and the in-flight state survives as a WIP snapshot
    wip = os.path.join(L.artifact_dir("t6"), "wip_context",
                       "interrupted_problem_01")
    snaps = [os.path.join(wip, d) for d in os.listdir(wip)]
    assert any("p_inflight" in open(os.path.join(s, L.LIBRARY_FILE)).read()
               for s in snaps)


def test_interleave_corpus_stable_prefix():
    basic = [f"b{i}" for i in range(12)]
    abstract = [f"a{i}" for i in range(3)]
    full = L.interleave_corpus(basic, abstract)
    assert len(full) == 15
    assert full[4] == "a0" and full[9] == "a1" and full[14] == "a2"
    # stable prefix: truncating the corpus never reorders earlier slots
    assert L.interleave_corpus(basic, abstract)[:8] == full[:8]


def test_infra_failure_waits_then_stops_resumably(sandbox):
    """Session-limit/credit-out guardrail: an infra failure must not consume
    ladder attempts; after max waits the run stops with no verdict recorded,
    library unchanged, so a relaunch resumes at the same problem."""
    calls = []

    def propose(task, ws, model, minutes):
        calls.append(model)
        with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
            f.write("def p_junk(panel):\n    return 0.0\n")
        return "You've hit your session limit - resets 12:50am"

    rep = L.run(_two_problems(), tag="t7", ws=str(sandbox / "ws7"),
                propose_fn=propose, verbose=False,
                infra_wait_seconds=0, max_infra_waits=2)
    # 1 first try + 2 retries after waits, all on rung 0, then stop
    assert calls == ["sonnet"] * 3
    assert rep.records == []  # no verdict recorded: not a solving failure
    lib = open(os.path.join(str(sandbox / "ws7"), L.LIBRARY_FILE)).read()
    assert "p_junk" not in lib


def test_infra_recovery_consumes_no_attempt(sandbox):
    """One infra failure, then a working proposer: the ladder still has all
    its rungs and the problem solves on attempt 1."""
    state = {"n": 0}

    def propose(task, ws, model, minutes):
        state["n"] += 1
        if state["n"] == 1:
            return "rate limit exceeded"
        with open(os.path.join(ws, L.LIBRARY_FILE), "a") as f:
            f.write(WITNESS)
        return "done"

    rep = L.run([two_vs_one_problem()], tag="t8", ws=str(sandbox / "ws8"),
                propose_fn=propose, verbose=False, infra_wait_seconds=0)
    assert rep.solved == 1
    assert rep.records[0].attempts == 1
    assert not rep.records[0].escalated
