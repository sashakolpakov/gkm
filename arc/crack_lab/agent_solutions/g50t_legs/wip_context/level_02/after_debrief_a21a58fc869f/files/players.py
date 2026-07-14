# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # g50t level 1: move the avatar to the goal region, using USE to open the
    # gate that unlocks the far side of the maze.  Fully handled by the general
    # unlock-then-reach leg.
    solve_unlock_reach(env, max_toggles=2)


def play_level_2(env):
    # g50t level 2: same "reach the goal chamber" objective, but the avatar's
    # region is only linked to the chamber through gates that open remotely when
    # the avatar walks over wall segments; a USE resets the avatar to start
    # while keeping opened gates.  A single USE does not enlarge reachability, so
    # the plain unlock leg fails here -- the general USE-macro unlock search
    # chains several staged openings until the chamber is reachable.
    solve_unlock_macro(env, max_expand=400)
