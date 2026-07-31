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


def play_level_3(env):
    traverse_teleport_revealed_exit_chain(
        env,
        bridge_control=(51, 18),
        connector_control=(51, 27),
        teleport_control=(51, 36),
        revealed_exit_control=(51, 45),
        initial_approach_segments=((3, 1),),
        first_crossing_segments=((3, 3), (2, 3), (3, 1)),
        connector_entry_segments=((3, 4),),
        teleport_approach_segments=((3, 3),),
        endpoint_segments=((1, 4),),
        activation_segments=((1, 2), (4, 2)),
        exit_staging_segments=((1, 1), (4, 1)),
        exit_segments=((1, 3), (4, 6), (2, 2), (4, 1)),
    )


def play_level_4(env):
    traverse_synchronized_builder_teleport_chain(
        env,
        bridge_control=(56, 28),
        shuttle_control=(46, 28),
        teleport_control=(52, 19),
        initial_approach_segments=((2, 1), (4, 1)),
        upper_build_direction=4,
        upper_build_phases=4,
        upper_crossing_segments=((4, 5), (2, 4), (4, 1)),
        bridge_restore_phases=4,
        shuttle_completion_phases=5,
        shuttle_crossing_segments=((3, 4),),
        teleport_entry_segments=((2, 2),),
        shuttle_reset_phases=1,
        lower_approach_segments=((2, 1), (4, 2), (2, 6)),
        lower_build_direction=4,
        lower_build_phases=4,
        exit_segments=((2, 1), (4, 6), (2, 2), (4, 1)),
    )
