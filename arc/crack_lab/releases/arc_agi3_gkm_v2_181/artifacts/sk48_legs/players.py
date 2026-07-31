# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    # Collect the requested 8 first.
    move_vertical_lanes(env, UP, 3)
    extend_tether(env, 4)
    retract_tether(env, 4)

    # Carry 8 to 14 and add it behind 8.
    move_vertical_lanes(env, DOWN, 2)
    extend_tether(env, 4)
    retract_tether(env, 3)

    # Carry the ordered pair to 9; contact completes the requested train.
    move_vertical_lanes(env, UP, 2)
    move_vertical_lanes(env, DOWN, 1)
    extend_tether(env, 3)


def play_level_2(env):
    reverse_row_train(
        env,
        approach_lanes=4,
        stages=((5, 6, 6), (4, 6, 5), (2, 5, 4)),
        final_extension=4,
    )


def play_level_3(env):
    weave_vertical_four_train(env)


def play_level_4(env):
    unweave_horizontal_pairs_to_vertical_heads(env)


def play_level_5(env):
    interleave_split_rows_compact(env)


def play_level_6(env):
    deal_and_shuttle_cross_collectors_compact(env)


def play_level_7(env):
    weave_shared_center_cross_compact(env)


def play_level_8(env):
    weave_cardinal_cross_with_shelf_handoffs(env)
