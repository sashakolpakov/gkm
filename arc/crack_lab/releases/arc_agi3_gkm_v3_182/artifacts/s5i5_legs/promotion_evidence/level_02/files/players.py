# Per-level players only compose reusable legs.
from legs import align_marked_bar_tips, route_telescoping_arm_through_gap


def play_level_1(env):
    align_marked_bar_tips(env)


def play_level_2(env):
    route_telescoping_arm_through_gap(env)
