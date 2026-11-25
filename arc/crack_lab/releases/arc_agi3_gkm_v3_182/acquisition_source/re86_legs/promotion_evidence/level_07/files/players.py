# Per-level players only compose reusable legs.
from legs import (align_selected_cross, align_selected_outline,
                  select_next_cross, align_outline_group, recolor_selected_outline,
                  deform_selected_outline)


def play_level_1(env):
    align_selected_cross(env)
    select_next_cross(env)
    align_selected_cross(env)


def play_level_2(env):
    align_selected_outline(env)
    select_next_cross(env)
    align_selected_outline(env)
    select_next_cross(env)
    align_selected_cross(env)


def play_level_3(env):
    align_outline_group(env, count=3)


def play_level_4(env):
    recolor_selected_outline(env, 12)
    align_selected_outline(env)
    select_next_cross(env)
    recolor_selected_outline(env, 14)
    align_selected_outline(env)


def play_level_5(env):
    recolor_selected_outline(env, 9, return_to_start=True)
    select_next_cross(env)
    recolor_selected_outline(env, 8, return_to_start=True)
    select_next_cross(env)
    recolor_selected_outline(env, 9, return_to_start=True, retreat_column=33)
    select_next_cross(env)
    align_outline_group(env, count=2, selection_stride=2, require_reward=False,
                        horizontal_first=True)
    select_next_cross(env)
    select_next_cross(env)
    align_selected_outline(env)


def play_level_6(env):
    deform_selected_outline(env, ((1, 5), (4, 1)), 4, 3,
                            departure=((2, 6), (4, 9)))
    align_selected_outline(env)
    select_next_cross(env)
    deform_selected_outline(env, ((2, 11), (3, 6)), 1, 2,
                            departure=((4, 6), (1, 4)))
    deform_selected_outline(env, (), 3, 2, departure=((1, 7),))
    align_selected_outline(env)


def play_level_7(env):
    deform_selected_outline(env, ((1, 10), (4, 7)), 2, 3)
    recolor_selected_outline(env, 11,
                             route=((3, 2), (1, 2), (2, 2), (4, 6), (2, 7)))
    align_selected_outline(env)
    select_next_cross(env)
    deform_selected_outline(env, ((4, 4), (1, 3)), 1, 2,
                            departure=((4, 7), (1, 4)))
    recolor_selected_outline(env, 9,
                             route=((1, 6), (3, 15), (1, 1), (2, 5), (4, 13)))
    align_selected_outline(env)
    select_next_cross(env)
    deform_selected_outline(env, ((1, 4), (4, 4), (2, 1), (4, 1)), 1, 1,
                            departure=((3, 10), (1, 9), (4, 3)))
    recolor_selected_outline(env, 8, route=((1, 2), (4, 9), (2, 2), (3, 9)))
