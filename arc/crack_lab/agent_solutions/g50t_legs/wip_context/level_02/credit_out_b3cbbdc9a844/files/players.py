# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # g50t level 1: move the avatar to the goal region, using USE to open the
    # gate that unlocks the far side of the maze.  Fully handled by the general
    # unlock-then-reach leg.
    solve_unlock_reach(env, max_toggles=2)
