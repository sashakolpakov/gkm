"""Offline unit tests for the enforced leg-library orchestration (gkm_legs.py).

No LLM, no game: the proposer and verifier are mocked so the control loop and the
MARGINAL free-energy accounting can be checked without credits. The point being
tested is the load-bearing property: reusing a leg is free, so later levels that add
no new legs have lower marginal novelty than early rule-learning levels.
"""
import os
import re
import shutil
import gkm_legs as L


def test_loc_ignores_blanks_and_comments():
    assert L._loc("def f():\n    return 1\n") == 2
    assert L._loc("\n# comment\n   \n") == 0


def test_marginal_complexity_reuse_is_free():
    legs = "def a(env):\n    pass\n"
    # reused leg (legs unchanged) + a 1-line player -> marginal C counts only the player
    assert L.marginal_complexity(legs, legs, "", "play(env)\n") == 1
    # adding a new leg is paid for
    legs2 = legs + "def b(env):\n    pass\n"
    assert L.marginal_complexity(legs, legs2, "", "") == 2


def test_free_energy_rewards_levels_and_penalises_novelty():
    assert L.free_energy(3, 0) == -3.0
    assert L.free_energy(3, 100, lam=0.02) == -3.0 + 2.0
    # more levels for the same novelty is always lower F
    assert L.free_energy(4, 50) < L.free_energy(3, 50)


def test_orchestration_loop_with_mocks_shows_reuse_trend(tmp_path=None):
    """Drive the loop with a mock proposer that INVENTS legs on L1-L2 (learning the
    rules) and REUSES them on L3-L4 (no new legs). Marginal novelty must drop."""
    def mock_propose(ws, K):
        with open(os.path.join(ws, "players.py"), "a") as f:
            f.write(f"\n\ndef play_level_{K}(env):\n    leg_1(env)\n    leg_2(env) if {K} > 1 else None\n")
        if K <= 2:  # early levels: invent a new leg
            with open(os.path.join(ws, "legs.py"), "a") as f:
                f.write(f"\n\ndef leg_{K}(env):\n    for _ in range(3):\n        pass\n")

    def mock_verify(game, solve_path):
        players = open(os.path.join(os.path.dirname(solve_path), "players.py")).read()
        n = len(re.findall(r"def play_level_\d+", players))
        return (n, [], None)   # empty path => real A.validate is skipped

    # isolated, clean workspace (fake game name; mocks ignore the real game)
    shutil.rmtree(os.path.join(L.SCRATCH, "gkm_legs_ws_legstest"), ignore_errors=True)
    rep = L.orchestrate("legstest", max_level=4, propose_fn=mock_propose,
                        verify_fn=mock_verify, debrief_fn=lambda w, k: None,
                        verbose=False)
    assert rep.reached == 4
    by = {r.level: r.marginal_C for r in rep.records}
    assert set(by) == {1, 2, 3, 4}
    # L1/L2 invent a leg (legs delta > 0); L3/L4 reuse (legs delta 0) -> strictly cheaper
    assert by[3] < by[1] and by[4] < by[2]
    assert by[3] == by[4]                       # pure reuse: identical marginal cost
    assert rep.total_marginal_C == sum(by.values())


def test_setup_workspace_builds_valid_dispatch():
    ws = L.setup_workspace("wa30")
    for f in ("legs.py", "players.py", "solve.py", "gkm_try.py", "legs_log.md"):
        assert os.path.exists(os.path.join(ws, f))
    import ast
    ast.parse(open(os.path.join(ws, "solve.py")).read())   # solve.py is valid Python
