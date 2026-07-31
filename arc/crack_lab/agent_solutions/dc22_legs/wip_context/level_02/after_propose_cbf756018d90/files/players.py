# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    traverse_two_stage_bridge_chain(
        env,
        lower_control=(48, 36),
        upper_control=(48, 19),
        entry_segments=((1, 4),),
        pivot_segments=((4, 5),),
        exit_segments=((1, 6), (4, 2)),
    )


def play_level_2(env):
    traverse_revealed_three_stage_bridge_chain(
        env,
        first_control=(52, 40),
        second_control=(52, 22),
        revealed_control=(52, 31),
        first_pivot_segments=((2, 5), (4, 5)),
        reveal_segments=((2, 6),),
        retreat_segments=((1, 6),),
        return_to_first_segments=((3, 5), (1, 5)),
        final_pivot_segments=((1, 1), (4, 1), (1, 2), (4, 7)),
        exit_segments=((1, 6),),
    )
