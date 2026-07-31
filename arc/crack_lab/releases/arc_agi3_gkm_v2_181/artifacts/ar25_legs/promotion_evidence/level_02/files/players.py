# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    repeat_action(env, 2, 10)
    repeat_action(env, 3, 5)


def play_level_2(env):
    repeat_action(env, 3, 2)
    repeat_action(env, 5, 1)
    repeat_action(env, 2, 8)
