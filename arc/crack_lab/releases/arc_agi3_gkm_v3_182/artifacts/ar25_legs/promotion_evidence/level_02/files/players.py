# Per-level players only compose reusable legs.
from legs import guide_colour_to_marker, switch_control


def play_level_1(env):
    guide_colour_to_marker(env, moving_colour=4, marker_colour=11)


def play_level_2(env):
    guide_colour_to_marker(env, moving_colour=4, marker_colour=11)
    switch_control(env)
    guide_colour_to_marker(env, moving_colour=4, marker_colour=11)
