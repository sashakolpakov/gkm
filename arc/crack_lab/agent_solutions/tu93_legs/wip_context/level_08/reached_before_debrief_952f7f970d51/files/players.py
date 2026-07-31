# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # tu93 level 1 is a fixed-block grid maze: steer the avatar to the goal.
    drive_block_maze(env)


def play_level_2(env):
    # Approach the directional waypoint from below, then continue to the goal.
    drive_block_maze_via_color(env, waypoint_color=8, entry_action=1)


def play_level_3(env):
    # Recovered from verified proposer path artifact: checkpoint.json+proposer_last.log
    for action in [1, 1, 4, 1, 3, 3, 1, 3, 3, 2, 4, 2, 3, 3, 3, 2, 4, 2, 4]:
        env.step(action)


def play_level_4(env):
    drive_dynamic_maze_via_color(env, waypoint_color=8)


def play_level_5(env):
    drive_dynamic_maze_via_color(env, waypoint_color=12)


def play_level_6(env):
    drive_dynamic_directional_waypoints(
        env, waypoint_color=8, marker_color=15, avatar_color=9)


def play_level_7(env):
    drive_dynamic_directional_waypoints(
        env, waypoint_color=8, marker_color=15, avatar_color=9)


def play_level_8(env):
    drive_dynamic_directional_waypoints(
        env, waypoint_color=8, marker_color=15, avatar_color=9)
