# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    move_vessel_below_and_apply(env)


def play_level_2(env):
    apply_current_then_select_and_apply_southeast(env, 46, 4)
