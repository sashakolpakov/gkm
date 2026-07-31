# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    copy_visible_color_code(env)


def play_level_2(env):
    copy_visible_color_code_into_diagram(env)


def play_level_3(env):
    copy_visible_color_code_into_diagram(env)


def play_level_4(env):
    copy_visible_color_code_into_diagram(env)


def play_level_5(env):
    copy_repeated_code_into_nested_diagram(env)


def play_level_6(env):
    factor_visible_code_into_parallel_diagrams(env)
