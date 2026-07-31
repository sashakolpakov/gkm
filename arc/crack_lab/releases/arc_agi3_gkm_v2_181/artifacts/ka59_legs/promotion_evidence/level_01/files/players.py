# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Push the initially selected ring through the central barrier.
    move_steps(env, RIGHT, 3)

    # Select the transferred ring and center it on the right target.
    select_at(env, 43, 31)
    move_steps(env, UP, 1)
    move_steps(env, RIGHT, 1)

    # Reselect the left ring and center it on the left target.
    select_at(env, 25, 31)
    move_steps(env, LEFT, 4)
    move_steps(env, DOWN, 1)
