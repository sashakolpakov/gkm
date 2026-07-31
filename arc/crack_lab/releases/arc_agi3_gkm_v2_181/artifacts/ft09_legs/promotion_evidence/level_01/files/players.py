# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Level 1 is a toggle-tile board: click a subset of tiles to reach the
    # hidden target configuration that completes the level.
    solve_toggle_board(env)
