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
    recolor_selected_outline(env, 12, route=((1, 5), (3, 8), (2, 3), (3, 5)))
    select_next_cross(env)
    recolor_selected_outline(env, 14, route=((2, 8), (4, 6), (1, 5), (3, 1)))


def play_level_5(env):
    recolor_selected_outline(env, 9, route=((3, 4), (4, 6)))
    select_next_cross(env)
    recolor_selected_outline(env, 8, route=((2, 10), (4, 7), (1, 4)))
    select_next_cross(env)
    recolor_selected_outline(env, 9, route=((2, 7), (3, 11), (4, 4), (1, 1)))
    select_next_cross(env)
    align_outline_group(env, count=2, selection_stride=2, require_reward=False,
                        horizontal_first=True)


def play_level_6(env):
    deform_selected_outline(env, ((1, 2), (4, 1)), 4, 3,
                            departure=((2, 3), (4, 9)))
    align_selected_outline(env)
    select_next_cross(env)
    deform_selected_outline(env, ((2, 11), (3, 5)), 1, 2,
                            departure=((4, 5), (1, 4)))
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


def play_level_8(env):
    deform_selected_outline(env, ((3, 7),), 2, 2, departure=((3, 1),))
    deform_selected_outline(env, (), 2, 1)
    recolor_selected_outline(env, 11,
                             route=((1, 4), (3, 2), (1, 10), (3, 1), (2, 2)))
    deform_selected_outline(env, ((4, 9), (1, 4)), 4, 3, departure=((2, 1),))
    deform_selected_outline(env, (), 4, 4,
                            departure=((3, 3), (2, 12), (3, 3)))
    deform_selected_outline(env, (), 2, 1, departure=((3, 3),))
    align_selected_outline(env)
    select_next_cross(env)
    deform_selected_outline(env, ((3, 6), (2, 1)), 2, 3)
    recolor_selected_outline(env, 6, route=((1, 10), (3, 6), (1, 2)))
    deform_selected_outline(env, ((4, 10), (1, 5)), 4, 6,
                            departure=((3, 3), (2, 14), (3, 3)))
    deform_selected_outline(env, (), 2, 4, departure=((3, 3),))
    align_selected_outline(env)
