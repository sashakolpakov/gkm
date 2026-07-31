# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    move_vessel_below_and_apply(env)


def play_level_2(env):
    apply_current_then_select_and_apply_southeast(env, 46, 4)


def play_level_3(env):
    apply_west_north_east_north_layers_then_payload(
        env,
        west_selector=52,
        north_selector=28,
        east_selector=46,
        payload_selector=34,
        selector_y=4,
        payload_x=31,
        payload_y=20,
    )


def play_level_4(env):
    apply_northwest_southeast_west_layers_then_west_payload(
        env,
        northwest_selector=34,
        southeast_selector=28,
        west_selector=58,
        payload_selector=40,
        selector_y=4,
        payload_x=14,
        payload_y=38,
    )
