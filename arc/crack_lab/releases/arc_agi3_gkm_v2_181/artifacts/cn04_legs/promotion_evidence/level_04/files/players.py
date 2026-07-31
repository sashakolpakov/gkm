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


def play_level_3(env):
    # Recovered from verified proposer path artifact: checkpoint.json+proposer_last.log
    for action in [[6, 35, 14], 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, [6, 35, 38], 5, 5, 3, 3, 3, 1, 1, 1, 1, 1, 1]:
        env.step(action)


def play_level_4(env):
    select_figure(env, 36, 21)
    rotate_quarter_turns(env, 1)
    walk(env, 2, 2)
    walk(env, 3, 7)

    select_figure(env, 45, 42)
    rotate_quarter_turns(env, 3)
    walk(env, 1, 4)
    walk(env, 3, 12)

    select_figure(env, 18, 48)
    rotate_quarter_turns(env, 1)
    walk(env, 1, 5)
    walk(env, 3, 3)
