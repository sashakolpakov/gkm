"""Test direct placement of the second remote cargo on level 9."""

import gkm_try

from perception import arr
from probe9_verify import (
    ReachedLevel9,
    StopAtLevel9,
    boxes,
    summary,
    target_state,
)


def shortest_dismiss(env, max_depth=6):
    frontier = [(env.clone(), [])]
    for depth in range(1, max_depth + 1):
        next_states = {}
        for node, path in frontier:
            for action in node.actions:
                child = node.clone()
                child.step(action)
                child_path = path + [action]
                if not boxes(child.frame(), 15):
                    return child, child_path
                next_states.setdefault(
                    arr(child.frame()).tobytes(),
                    (child, child_path),
                )
        frontier = list(next_states.values())
        print("ALT_DISMISS_DEPTH", depth, len(frontier), flush=True)
    return None, None


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    remote_pick = [2] + [4] * 6 + [1, 5, 2]
    first_delivery = remote_pick + [3] * 6 + [1] * 3 + [3, 5]
    second_pick = first_delivery + [2] * 4 + [4] * 5 + [1] * 2 + [5]
    direct_bottom_middle = [2] + [3] * 6 + [1] * 3 + [5, 2]

    state = env.clone()
    for action in second_pick + direct_bottom_middle:
        state.step(action)
    turn = len(second_pick) + len(direct_bottom_middle)
    print("ALT_DIRECT", summary(state, turn), flush=True)

    dismissed, contact = shortest_dismiss(state)
    print(
        "ALT_DISMISS",
        contact,
        None if dismissed is None else summary(
            dismissed, turn + len(contact)
        ),
        flush=True,
    )
    if dismissed is None:
        return

    idle = dismissed.clone()
    idle_turn = turn + len(contact)
    previous = None
    while not idle.terminal():
        target = target_state(idle.frame())
        condensed = (target["empty"], target["filled"])
        if condensed != previous or idle_turn % 4 == 0:
            print(
                "ALT_IDLE",
                idle_turn,
                condensed,
                boxes(idle.frame(), 12),
                boxes(idle.frame(), 4),
                flush=True,
            )
        previous = condensed
        idle.step(5)
        idle_turn += 1
    print("ALT_TERMINAL", idle_turn, summary(idle, idle_turn), flush=True)


gkm_try.A.run_program("wa30", inspect)
