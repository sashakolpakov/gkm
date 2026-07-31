# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    follow_diagonal_lattice_to_ring(env)


def play_level_2(env):
    merge_equal_squares_and_deliver_to_ring(env)


def play_level_3(env):
    merge_equal_squares_and_deliver_to_ring(env)
    deliver_remaining_square_via_diagonal_detour(env)


def play_level_4(env):
    merge_equal_squares_around_moving_cutter(env)
