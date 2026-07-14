# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Level 1 is a tile-cycle matching puzzle: cycle each tile's glyph to a
    # hidden target. Reuse the general tile-cycle solver leg.
    solve_tile_cycle_puzzle(env)


def play_level_2(env):
    # Level 2 is a legend-decode variant of the tile puzzle: a coded 'word'
    # (wide reference box) is translated through a key->value legend into the
    # sequence of glyphs the editable tiles must show. Reward is opaque, so we
    # decode the target from the frame and drive each tile to it directly.
    solve_glyph_cipher_puzzle(env)
