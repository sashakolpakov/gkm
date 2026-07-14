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


def play_level_3(env):
    # Level 3 is a richer legend cipher: keys AND values are bordered boxes in
    # key->value pairs, and a single key may span several glyphs, so the coded
    # 'word' (the one unpaired key-colored box) must be SEGMENTED (tokenised)
    # into keys before substituting each by its value glyph-sequence. Reuse the
    # general legend-segmentation solver leg.
    solve_glyph_legend_puzzle(env)


def play_level_4(env):
    # Level 4 is a COMPOSED cipher: two legends chained by border colour. A
    # source word (border SRC) is translated legend-by-legend (SRC->..->VAL,
    # following each pair's key-border -> value-border) into the target glyphs
    # the editable word's tiles (border VAL) must show. Reuse the general
    # multi-legend composition solver leg.
    solve_glyph_compose_puzzle(env)
