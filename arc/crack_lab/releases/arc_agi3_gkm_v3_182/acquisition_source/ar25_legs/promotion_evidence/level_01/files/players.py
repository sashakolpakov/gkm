# Per-level players only compose reusable legs.
from legs import guide_colour_to_marker


def play_level_1(env):
    guide_colour_to_marker(env, moving_colour=4, marker_colour=11)
