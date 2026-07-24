# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    follow_action_sequence(env, MIRRORED_PAIR_ASCENT)


def play_level_2(env):
    follow_action_sequence(env, MIRRORED_PAIR_MAZE_REUNION)
