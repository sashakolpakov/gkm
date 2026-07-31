"""Test filling middle-right before bottom-right on level 9."""

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
    return None, None


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    middle_right = (
        [4] * 5 + [1, 3, 5]
        + [2] * 2 + [3] * 5 + [1] * 5
        + [5, 2]
    )
    bottom_right = (
        [2] * 4 + [4] * 6 + [1, 5]
        + [3] * 7 + [1] * 3 + [5, 2]
    )
    state = env.clone()
    for action in middle_right:
        state.step(action)
    print("REVERSE_MIDDLE", summary(state, len(middle_right)), flush=True)
    for action in bottom_right:
        state.step(action)
    turn = len(middle_right) + len(bottom_right)
    print("REVERSE_BOTTOM", summary(state, turn), flush=True)

    dismissed, contact = shortest_dismiss(state)
    print(
        "REVERSE_DISMISS",
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
    prior = None
    while not idle.terminal():
        target = target_state(idle.frame())
        condensed = (target["empty"], target["filled"])
        if condensed != prior:
            print("REVERSE_IDLE", idle_turn, condensed, flush=True)
        prior = condensed
        idle.step(5)
        idle_turn += 1
    print("REVERSE_TERMINAL", idle_turn, summary(idle, idle_turn),
          flush=True)


gkm_try.A.run_program("wa30", inspect)
