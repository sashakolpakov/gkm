# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    make_small_segments_color_5(env)
    click_largest_color_9_submit_disc(env)


def play_level_2(env):
    frame = turn_on_outer_rows_of_right_segment_panel(env)
    if frame is not None:
        click_largest_color_9_submit_disc(env, frame)


def play_level_3(env):
    configure_then_submit(env, encode_reacquisition_route_through_barrier)


def play_level_4(env):
    configure_then_submit(env, reacquire_max_scaled_agent_and_route_to_socket)


def play_level_5(env):
    configure_then_submit(env, encode_route_resize_rotate_and_acquire_socket)


def play_level_6(env):
    route_to_socket_via_reacquisition_checkpoints(env)
