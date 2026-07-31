# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    align_slider_tips_to_hollow_targets(env)


def play_level_2(env):
    move_articulated_marker_around_barrier(env)


def play_level_3(env):
    dock_crossed_sliders_through_coupled_barriers(env)


def play_level_4(env):
    extend_shared_marker_through_staged_crossbars(env)


def play_level_5(env):
    thread_coupled_marker_through_reconfigurable_frame(env)


def play_level_6(env):
    dock_three_link_arm_through_partitioned_chamber(env)
