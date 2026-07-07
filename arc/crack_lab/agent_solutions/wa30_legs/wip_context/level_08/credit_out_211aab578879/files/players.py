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
    # L8 is a two-arena box->socket sokoban with FOUR autonomous agents
    # (colour-12 couriers that seat boxes, colour-15 movers that steal them
    # back) under a hard ~150-step budget.  No agent can be frozen (L7) or
    # gap-cleared (L6), and seating never accumulates while the thieves patrol,
    # so the earlier neutralise-then-deliver legs do not apply.  Best-effort:
    # probe the general pixel box->socket controller on a CLONE and only commit
    # its actions if it actually clears the level (keeps the real path clean if
    # it does not).
    probe = env.clone()
    if deliver_boxes_to_sockets(probe, prefer='both'):
        for a in probe.path[len(env.path):]:
            if env.terminal():
                break
            env.step(a)


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
