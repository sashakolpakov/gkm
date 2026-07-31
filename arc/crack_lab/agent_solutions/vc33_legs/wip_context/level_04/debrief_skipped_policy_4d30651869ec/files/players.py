# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    align_lower_cyan_tip_with_upper_notch(env)


def play_level_2(env):
    align_lower_cyan_tip_with_upper_notch(env, marker_color=14)


def play_level_3(env):
    relay_height_between_adjacent_reservoirs(env)


def play_level_4(env):
    cross_pressure_gates_then_align_height(env)
