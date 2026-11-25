# Per-level players only compose reusable legs.
from legs import match_miniature


def play_level_1(env):
    match_miniature(env, 36, 36)


def play_level_2(env):
    match_miniature(env, 20, 14, additional_boards=((20, 30),))
