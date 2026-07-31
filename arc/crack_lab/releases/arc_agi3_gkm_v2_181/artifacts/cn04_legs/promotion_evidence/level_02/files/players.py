# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    walk(env, 2, 7)
    walk(env, 4, 4)
    rotate_quarter_turns(env, 3)


def play_level_2(env):
    walk(env, 1, 2)
    walk(env, 4, 4)

    select_figure(env, 23, 35)
    walk(env, 1, 8)
    walk(env, 4, 4)

    select_figure(env, 50, 53)
    rotate_quarter_turns(env, 3)
    walk(env, 1, 10)
