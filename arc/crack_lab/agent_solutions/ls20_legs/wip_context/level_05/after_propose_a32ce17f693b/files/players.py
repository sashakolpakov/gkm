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
    # Same sliding-5x5-tile-on-track mechanic again, plus the same drifting
    # HUD, plus extra moving pieces (a patrolling object, hidden switches)
    # that only enlarge the state space -- raw-frame BFS still captures all
    # of it, it just needs a bigger budget than level 4's maze.
    solve_masked(env, max_states=60000)
