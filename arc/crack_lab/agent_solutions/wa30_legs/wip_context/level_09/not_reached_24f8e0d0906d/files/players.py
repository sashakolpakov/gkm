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


def play_level_6(env):
    # A self-mover (colour 15) races through the wall gap, ferries the west
    # box into the east store and parks in the gap.  Thread the gap to it and
    # USE to clear it, then carry BOTH east boxes into the west goal container
    # (a filled container is the win).
    clear_agent_then_deliver(env,
                             [(UP, 7), (RIGHT, 6)],
                             [((6, 13), (4, 8)), ((7, 14), (4, 7))])


def play_level_7(env):
    # An uncatchable self-mover (colour 15) hauls every loose box to its own
    # store; USE can't clear it while it patrols.  Let it stall (grab a box it
    # can't seat), USE to clear it, then fill the west 9-framed goal container.
    clear_frozen_mover_then_fill(env, targets=[(8, 3), (7, 3)])


def play_level_8(env):
    # Two socket containers (9-frame/2-core), each with a courier (colour 12)
    # and a roaming same-speed STEALER (colour 15) that undoes the courier.
    # Corner-clear both stealers, then let the couriers + a few avatar
    # deliveries drain both sockets.
    top_cells = [(3, 12), (3, 13), (2, 12), (2, 13),
                 (3, 11), (2, 11), (3, 14), (2, 14)]
    bot_cells = [(13, 13), (13, 12), (12, 13), (12, 12),
                 (13, 14), (12, 14), (14, 13), (14, 12), (14, 14)]
    clear_stealers_then_fill_dual(
        env,
        bands=[(lambda m: m[0] < 7, (5, 5)),
               (lambda m: m[0] > 9, (14, 7))],
        top_cells=top_cells,
        bot_cells=bot_cells,
        split_row=9,
    )


def play_level_9(env):
    # Hard 70-move level clock; win = every box seated in a container.
    # Left: a courier auto-fills the big 3x3 container from the six west
    # boxes; a stealer snakes the bottom maze toward its single exit (9,3).
    # Right: two bottom slots take two boxes directly, and the third is
    # released ON the impassable diagonal band, where the parked top courier
    # fetches and seats it.  Then seal the maze exit to freeze the stealer,
    # corner-USE it, carry one west box ourselves (the courier alone is too
    # slow), and yield the last turns to the courier.
    deliver_pairs(env, [((8, 14), (6, 14)),
                        ((7, 12), (6, 13)),
                        ((5, 11), (3, 11))])
    pin_maze_mover(env, (9, 3))
    deliver_pairs(env, [((8, 1), (5, 6)), ((7, 1), (4, 6))])
    yield_until_level_up(env, USE, cap=40)


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
