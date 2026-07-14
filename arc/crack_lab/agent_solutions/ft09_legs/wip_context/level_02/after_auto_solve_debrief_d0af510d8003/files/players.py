# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Level 1 is a toggle-tile board (framed 3x3): click a subset of tiles to
    # reach the hidden target configuration that completes the level.
    solve_toggle_board(env)


def play_level_2(env):
    # Level 2 is the SAME toggle-tile board family, only a wider 3-row grid
    # (~13 tiles). No new skill is needed: the shared solve_toggle_board leg
    # probes for the tiles and subset-searches the completing clicks on clones.
    solve_toggle_board(env)
