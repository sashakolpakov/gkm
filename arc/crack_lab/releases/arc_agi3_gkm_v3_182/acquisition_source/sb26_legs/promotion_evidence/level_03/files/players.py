"""Level players compose reusable legs."""
from legs import match_colour_row, match_nested_colour_rows


def play_level_1(env):
    match_colour_row(env)


def play_level_2(env):
    match_nested_colour_rows(env)


def play_level_3(env):
    match_nested_colour_rows(env)
