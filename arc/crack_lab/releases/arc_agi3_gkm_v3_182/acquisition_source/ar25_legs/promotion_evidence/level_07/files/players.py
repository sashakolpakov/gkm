# Per-level players only compose reusable legs.
from legs import (guide_colour_to_marker, switch_control, align_reflection_axis,
                  guide_selected_shape_to_marker,
                  cover_markers_with_reflected_shapes)


def play_level_1(env):
    guide_colour_to_marker(env, moving_colour=4, marker_colour=11)


def play_level_2(env):
    guide_colour_to_marker(env, moving_colour=4, marker_colour=11)
    switch_control(env)
    guide_colour_to_marker(env, moving_colour=4, marker_colour=11)


def play_level_3(env):
    align_reflection_axis(env)
    switch_control(env)
    guide_selected_shape_to_marker(env)
    switch_control(env)
    guide_selected_shape_to_marker(env)


def play_level_4(env):
    align_reflection_axis(env)
    switch_control(env)
    guide_selected_shape_to_marker(env)
    switch_control(env)
    guide_selected_shape_to_marker(env)


def play_level_5(env):
    align_reflection_axis(env)
    switch_control(env)
    align_reflection_axis(env, axis=1)
    switch_control(env)
    guide_selected_shape_to_marker(env)


def play_level_6(env):
    align_reflection_axis(env)
    switch_control(env)
    align_reflection_axis(env, axis=1)
    switch_control(env)
    guide_selected_shape_to_marker(env)
    switch_control(env)
    guide_selected_shape_to_marker(env, motion_template=True)


def play_level_7(env):
    align_reflection_axis(env)
    switch_control(env)
    align_reflection_axis(env, axis=1)
    switch_control(env)
    cover_markers_with_reflected_shapes(env)
