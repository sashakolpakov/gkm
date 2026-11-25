# Per-level players only compose reusable legs.
from legs import (align_selected_cross, align_selected_outline,
                  select_next_cross, align_outline_group, recolor_selected_outline)


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
