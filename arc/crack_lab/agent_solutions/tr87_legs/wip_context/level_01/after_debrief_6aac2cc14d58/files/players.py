# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Level 1 is a tile-cycle matching puzzle: cycle each tile's glyph to a
    # hidden target. Reuse the general tile-cycle solver leg.
    solve_tile_cycle_puzzle(env)
