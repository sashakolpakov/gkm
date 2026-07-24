# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    repeat_action(env, 2, 10)
    repeat_action(env, 3, 5)


def play_level_2(env):
    repeat_action(env, 3, 2)
    repeat_action(env, 5, 1)
    repeat_action(env, 2, 8)


def play_level_3(env):
    # Select and align the L with its lower silhouette.
    repeat_action(env, 5, 1)
    repeat_action(env, 2, 7)
    repeat_action(env, 4, 7)

    # Select and align the T with its lower silhouette.
    repeat_action(env, 5, 1)
    repeat_action(env, 2, 5)
    repeat_action(env, 3, 12)

    # Return to the scanner and sweep upward to validate the matches.
    repeat_action(env, 5, 1)
    repeat_action(env, 1, 16)
