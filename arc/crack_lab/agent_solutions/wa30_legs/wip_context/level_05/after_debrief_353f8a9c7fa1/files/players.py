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


def play_level_5(env):
    # Avatar shuttles all six arena boxes to rotating courier-handoff cells,
    # then yields so the lone courier seats the last ones inside the budget.
    ferry_all_to_courier_then_yield(env, drops=((7, 10), (8, 10)))


def play_level_4(env):
    # Ferry the six fenced arena boxes into courier reach, then yield.
    ferry_all_then_yield(env, [
        ([(RIGHT, 1)], DOWN, DOWN, 2),
        ([(UP, 1), (RIGHT, 1)], DOWN, DOWN, 1),
        ([(LEFT, 2)], LEFT, LEFT, 1),
        ([(UP, 3), (RIGHT, 1), (UP, 1)], LEFT, LEFT, 1),
        ([(DOWN, 1), (RIGHT, 3)], UP, DOWN, 2),  # face up, pull down
        ([(LEFT, 1), (UP, 1)], RIGHT, RIGHT, 2),
        ([(UP, 1), (LEFT, 3), (UP, 1)], RIGHT, RIGHT, 3),
    ], DOWN)
