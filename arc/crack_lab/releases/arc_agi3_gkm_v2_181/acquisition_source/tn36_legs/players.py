# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    configure_then_submit(env, make_small_segments_color_5)


def play_level_2(env):
    configure_then_submit(
        env,
        turn_on_outer_rows_of_right_segment_panel,
        submit_captured_frame=True,
    )


def play_level_3(env):
    configure_then_submit(env, encode_reacquisition_route_through_barrier)


def play_level_4(env):
    configure_then_submit(env, reacquire_max_scaled_agent_and_route_to_socket)


def play_level_5(env):
    configure_then_submit(env, encode_route_resize_rotate_and_acquire_socket)


def play_level_6(env):
    execute_reacquisition_plan(
        env,
        build_checkpoint_socket_reacquisition_plan(env),
    )


def play_level_7(env):
    execute_reacquisition_plan(
        env,
        build_matching_socket_reacquisition_plan(env),
    )
