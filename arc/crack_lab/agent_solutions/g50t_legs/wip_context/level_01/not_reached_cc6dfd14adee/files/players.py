# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Level 1 is a maze/switch level. The reusable skill is a bounded clone
    # search for a path that completes the level.
    solve_level_by_search(env)
