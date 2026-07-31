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
