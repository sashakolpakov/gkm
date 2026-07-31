# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    align_slider_tips_to_hollow_targets(env)


def play_level_2(env):
    move_articulated_marker_around_barrier(env)
