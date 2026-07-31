# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    walk(env, 2, 7)
    walk(env, 4, 4)
    rotate_quarter_turns(env, 3)
