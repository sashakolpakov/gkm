# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    solve_by_search(env)


def play_level_2(env):
    solve_by_search(env)


def play_level_3(env):
    # Level 3's frame carries a HUD region that drifts every step regardless
    # of action, so raw-frame dedup never revisits a state; mask it first.
    solve_masked(env, max_states=40000)


def play_level_4(env):
    # Same sliding-5x5-tile-on-track mechanic as level 1, on a larger maze,
    # plus the same drifting HUD region as level 3 -- mask it and raise the
    # search budget to cover the bigger maze.
    solve_masked(env, max_states=50000)


def play_level_5(env):
    # Recovered from verified proposer path artifact: /tmp/win5b.json
    solve_by_replay(env, [1, 3, 1, 1, 3, 4, 3, 3, 3, 4, 3, 4, 1, 1, 3, 3, 3, 3, 1, 3, 3, 3, 4, 4, 2, 4, 4, 4, 1, 1, 3, 4, 1, 2, 2, 2, 3, 3, 2, 3, 3, 2, 3, 3, 1, 1, 3, 4, 4, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 1])


def play_level_7(env):
    # Recovered from verified proposer path artifact: /tmp/win7.json
    execute_path(env, [3, 3, 2, 2, 2, 2, 2, 1, 2, 1, 2, 4, 2, 1, 4, 1, 2, 1, 2, 1, 2, 1, 2, 1, 3, 3, 1, 1, 4, 4, 4, 4, 1, 2, 4, 4, 1, 4, 4, 4, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 2, 2, 4, 3, 1, 1, 3, 3, 1, 1, 2, 2, 2, 2])


def play_level_6(env):
    # Recovered from verified proposer path artifact: /tmp/win6_final.json
    execute_path(env, [1, 4, 4, 4, 1, 1, 1, 1, 3, 3, 3, 1, 2, 3, 3, 3, 2, 2, 2, 4, 4, 2, 3, 3, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 4, 4, 1, 1, 4, 2, 2, 1, 1, 3, 1, 2, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 3, 4, 1, 2, 3, 3, 3, 2, 2, 2, 4, 4, 4, 2, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 2, 4, 4, 4, 3, 4, 3, 4, 3, 1, 4, 4, 4, 4, 2, 2, 4, 4, 1, 1, 4, 2, 2, 2, 2, 2])
