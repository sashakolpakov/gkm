# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    make_small_segments_color_5_and_submit(env)


def play_level_2(env):
    turn_on_outer_rows_of_right_segment_panel_and_submit(env)
