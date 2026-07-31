# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    align_colored_crosses_to_ring_axes(env)


def play_level_2(env):
    align_selected_outlines_to_ring_markers(env)
