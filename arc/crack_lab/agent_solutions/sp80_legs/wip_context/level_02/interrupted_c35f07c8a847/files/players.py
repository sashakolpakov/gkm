# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    """Level 1: move the 9-block right 3 times then USE."""
    bfs_or_fallback(env, make_bbox_key(9), fallback=[4, 4, 4, 5],
                    actions=(1, 2, 3, 4, 5, 6), max_states=500, max_depth=20)
