# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Level 1 is a tile-cycle matching puzzle: cycle each tile's glyph to a
    # hidden target. Reuse the general tile-cycle solver leg.
    solve_tile_cycle_puzzle(env)


def play_level_2(env):
    # Level 2 is a legend-decode variant of the tile puzzle: a coded 'word'
    # (wide reference box) is translated through a key->value legend into the
    # sequence of glyphs the editable tiles must show. Same perceive->decode->
    # plan->replay pipeline as L3/L4 (solve_glyph_cipher_via); this level's only
    # difference is its decoder, discover_glyph_cipher_puzzle.
    solve_glyph_cipher_via(env, discover_glyph_cipher_puzzle)


def play_level_3(env):
    # Level 3 is a richer legend cipher: keys AND values are bordered boxes in
    # key->value pairs, and a single key may span several glyphs, so the coded
    # 'word' (the one unpaired key-colored box) must be SEGMENTED (tokenised)
    # into keys before substituting each by its value glyph-sequence. Same
    # pipeline as L2/L4, differing only in its decoder.
    solve_glyph_cipher_via(env, discover_glyph_legend_puzzle)


def play_level_4(env):
    # Level 4 is a COMPOSED cipher: two legends chained by border colour. A
    # source word (border SRC) is translated legend-by-legend (SRC->..->VAL,
    # following each pair's key-border -> value-border) into the target glyphs
    # the editable word's tiles (border VAL) must show. Same pipeline as L2/L3,
    # differing only in its decoder.
    solve_glyph_cipher_via(env, discover_glyph_compose_puzzle)
