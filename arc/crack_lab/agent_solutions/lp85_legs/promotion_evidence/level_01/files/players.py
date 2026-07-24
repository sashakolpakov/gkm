# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa
from legs import repeat_click


def play_level_1(env):
    repeat_click(env, 5, 32, 5)
