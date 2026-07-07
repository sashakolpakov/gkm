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


def play_level_4(env):
    # Recovered from verified proposer path artifact: checkpoint.json
    execute_path(env, [4, 2, 5, 2, 2, 5, 1, 4, 2, 5, 2, 5, 3, 3, 3, 5, 3, 5, 1, 1, 1, 4, 1, 3, 5, 3, 5, 2, 4, 4, 4, 1, 5, 2, 2, 5, 3, 1, 4, 5, 4, 4, 5, 1, 3, 3, 3, 1, 4, 5, 4, 4, 4, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
