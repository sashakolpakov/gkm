# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa
from legs import repeat_click


def play_level_1(env):
    repeat_click(env, 5, 32, 5)


def play_level_2(env):
    repeat_click(env, 39, 17, 1)
    repeat_click(env, 48, 35, 1)
    repeat_click(env, 39, 17, 3)
    repeat_click(env, 48, 35, 3)


def play_level_3(env):
    repeat_click(env, 23, 41, 4)
    repeat_click(env, 35, 41, 4)
    repeat_click(env, 23, 41, 6)
    repeat_click(env, 35, 41, 2)
