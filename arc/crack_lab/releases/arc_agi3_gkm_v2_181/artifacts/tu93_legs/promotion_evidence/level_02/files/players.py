# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # tu93 level 1 is a fixed-block grid maze: steer the avatar to the goal.
    drive_block_maze(env)


def play_level_2(env):
    # Approach the directional waypoint from below, then continue to the goal.
    drive_block_maze_via_color(env, waypoint_color=8, entry_action=1)
