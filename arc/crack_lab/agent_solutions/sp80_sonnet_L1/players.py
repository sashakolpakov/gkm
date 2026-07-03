# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    sense_align_trigger(env, best_deflect_left_col, move_bar_to_left_col, pour)
