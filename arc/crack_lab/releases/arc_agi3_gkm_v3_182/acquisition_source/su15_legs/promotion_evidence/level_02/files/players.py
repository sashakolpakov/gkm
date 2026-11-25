# Each level player composes reusable legs.
from legs import click_walk_to_region, merge_matching_markers


def play_level_1(env):
    click_walk_to_region(env)


def play_level_2(env):
    merge_matching_markers(env)
    click_walk_to_region(env, marker_color=11, stride=3)
