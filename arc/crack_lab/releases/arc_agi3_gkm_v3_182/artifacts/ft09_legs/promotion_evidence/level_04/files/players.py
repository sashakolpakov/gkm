# Per-level players only compose reusable legs.
from legs import match_miniature


def play_level_1(env):
    match_miniature(env, 36, 36)


def play_level_2(env):
    match_miniature(env, 20, 14, additional_boards=((20, 30),))


def play_level_3(env):
    match_miniature(env, 20, 4, tile_colors=(8, 12),
                    additional_boards=((12, 20), (28, 20), (20, 36)))


def play_level_4(env):
    match_miniature(env, 12, 14, tile_colors=(9, 8, 12),
                    additional_boards=((28, 14), (20, 30)))
