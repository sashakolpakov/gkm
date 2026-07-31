# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # tu93 level 1 is a fixed-block grid maze: steer the avatar to the goal.
    drive_block_maze(env)
