# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Collect the requested 8 first.
    move_vertical_lanes(env, UP, 3)
    extend_tether(env, 4)
    retract_tether(env, 4)

    # Carry 8 to 14 and add it behind 8.
    move_vertical_lanes(env, DOWN, 2)
    extend_tether(env, 4)
    retract_tether(env, 3)

    # Carry the ordered pair to 9; contact completes the requested train.
    move_vertical_lanes(env, UP, 2)
    move_vertical_lanes(env, DOWN, 1)
    extend_tether(env, 3)
