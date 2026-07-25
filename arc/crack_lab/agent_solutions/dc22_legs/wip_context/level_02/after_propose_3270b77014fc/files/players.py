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
