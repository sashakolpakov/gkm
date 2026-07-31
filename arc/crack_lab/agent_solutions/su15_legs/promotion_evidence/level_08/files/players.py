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


def play_level_5(env):
    merge_equal_squares_around_moving_cutter(
        env, max_depth=36, minimum_stage_mass=2
    )


def play_level_6(env):
    stage_large_square_for_diagonal_partner(env)


def play_level_7(env):
    merge_equal_squares_around_moving_cutter(env)


def play_level_8(env):
    targets = corner_ring_targets_for_coupled_layout(env)
    if targets is None:
        return
    upper_left, lower_left = targets
    if not merge_small_squares_along_corner_lane(env, lower_left):
        return
    if not move_solid_square_to_target(env, 12, lower_left, max_moves=1):
        return
    if not move_solid_square_to_target(env, 8, upper_left, max_moves=2):
        return
    if not merge_moving_bodies_preserving_cutter(env):
        return
    if not reseat_square_while_cutting_staged_square(env, lower_left):
        return
    route_cutter_and_merged_body_to_corner_rings(env)
