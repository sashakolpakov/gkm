# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    bfs_to_level_up(env)


def play_level_2(env):
    bfs_to_level_up(env)


def play_level_3(env):
    bfs_to_level_up(env)


def play_level_4(env):
    bfs_to_level_up(env, max_nodes=30000)
