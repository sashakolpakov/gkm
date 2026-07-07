# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    solve_by_search(env)


def play_level_2(env):
    solve_by_search(env)


def play_level_3(env):
    # Level 3's frame carries a HUD region that drifts every step regardless
    # of action, so raw-frame dedup never revisits a state; mask it first.
    mask = detect_noise_mask(env)
    solve_by_search(env, max_states=40000, mask=mask)


def play_level_4(env):
    # Same sliding-5x5-tile-on-track mechanic as level 1, on a larger maze,
    # plus the same drifting HUD region as level 3 -- mask it and raise the
    # search budget to cover the bigger maze.
    mask = detect_noise_mask(env)
    solve_by_search(env, max_states=50000, mask=mask)
