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


def play_level_5(env):
    cross_horizontal_gates_then_align_opposing_markers(env)


def play_level_6(env):
    cross_pressure_gates_then_align_height(env)
    align_marker_pair_with_pressure_controls(env)


def play_level_7(env):
    cross_pressure_gate_toward_matching_marker(env, marker_color=15)
    align_marker_pair_with_pressure_controls(
        env,
        marker_color=15,
        max_stages=24,
        max_states=1200,
        max_depth=18,
    )
    cross_pressure_gates_then_align_height(
        env,
        marker_color=11,
        max_stages=16,
        max_states=1000,
        max_depth=20,
    )
    cross_pressure_gates_toward_matching_markers(
        env, marker_colors=(11, 14)
    )
    balance_marked_reservoirs_through_center(env)
