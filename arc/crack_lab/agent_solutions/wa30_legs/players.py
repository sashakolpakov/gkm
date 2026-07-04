# Per-level players. Each play_level_K(env) ONLY composes legs from legs.py.
from legs import *  # noqa


def play_level_1(env):
    ascend_and_use(env, 3)
    ascend_and_use(env, 2)
    service_left_branch(env)
    service_right_branch(env)


def play_level_2(env):
    enter_lower_right_lane(env)
    sweep_to_lower_left_switch(env)
    sweep_to_far_right_switch(env)
    reset_on_lower_left_switch(env)
    nudge_to_center_lift(env)
    finish_right_exit(env)


def play_level_3(env):
    follow_plan(env, (LEFT, 3), (DOWN, 2), (RIGHT, 1))
    relay_box_from_west(env, 5)

    follow_plan(env, (LEFT, 4), (UP, 6), (RIGHT, 1))
    relay_box_from_west(env, 3)

    follow_plan(env, (LEFT, 7), (UP, 1), (RIGHT, 1))
    relay_box_from_west(env, 6, depart_action=LEFT)

    use(env, 33)
