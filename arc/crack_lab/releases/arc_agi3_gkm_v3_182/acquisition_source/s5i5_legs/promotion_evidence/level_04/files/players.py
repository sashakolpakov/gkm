# Per-level players only compose reusable legs.
from legs import (align_marked_bar_tips, route_telescoping_arm_through_gap,
                  park_crossing_telescopic_barriers)


def play_level_1(env):
    align_marked_bar_tips(env)


def play_level_2(env):
    route_telescoping_arm_through_gap(env)


def play_level_3(env):
    park_crossing_telescopic_barriers(env)
    align_marked_bar_tips(env)


def play_level_4(env):
    park_crossing_telescopic_barriers(
        env, stages=((14, 1, 4), (9, 1, 4), (12, 0, 7), (8, 1, 4)))
    align_marked_bar_tips(env)
