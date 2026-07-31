# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    follow_cardinal_runs(
        env,
        (
            (LEFT, 3),
            (UP, 4),
            (RIGHT, 3),
            (UP, 3),
        ),
    )


def play_level_2(env):
    follow_cardinal_runs(
        env,
        (
            (UP, 1),
            (RIGHT, 1),
            (UP, 5),
            (RIGHT, 3),
            (DOWN, 8),
            (LEFT, 2),
            (RIGHT, 2),
            (UP, 2),
            (DOWN, 1),
            (UP, 7),
            (LEFT, 7),
            (DOWN, 6),
        ),
    )


def play_level_3(env):
    follow_cardinal_runs(
        env,
        (
            (UP, 8),
            (LEFT, 1),
            (DOWN, 2),
            (RIGHT, 1),
            (LEFT, 1),
            (DOWN, 5),
            (LEFT, 1),
            (DOWN, 1),
            (RIGHT, 1),
            (UP, 3),
            (LEFT, 2),
            (UP, 1),
            (RIGHT, 7),
            (UP, 3),
            (LEFT, 1),
            (RIGHT, 1),
            (LEFT, 1),
            (RIGHT, 1),
            (UP, 1),
            (DOWN, 1),
        ),
    )
