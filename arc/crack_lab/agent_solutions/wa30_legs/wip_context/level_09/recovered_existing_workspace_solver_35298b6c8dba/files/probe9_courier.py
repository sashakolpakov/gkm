"""Locate the first courier interference in the fast level-9 route."""

import gkm_try

from probe9_candidate import direct_second_prefix
from probe9_verify import (
    ReachedLevel9,
    StopAtLevel9,
    boxes,
    target_state,
)


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    prefix = direct_second_prefix() + [2, 2, 3, 5]
    state = env.clone()
    for action in prefix:
        state.step(action)

    local_actions = (
        [3] * 4 + [5, 4]
        + [1] * 6 + [4] * 5 + [2] * 2 + [5, 1]
        + [5] * 5
    )
    active = state.clone()
    idle = state.clone()
    for offset, action in enumerate(local_actions, 1):
        if active.terminal() or idle.terminal():
            break
        active.step(action)
        idle.step(5)
        turn = len(prefix) + offset
        active_courier = boxes(active.frame(), 12)
        idle_courier = boxes(idle.frame(), 12)
        active_target = target_state(active.frame())
        idle_target = target_state(idle.frame())
        print(
            "COURIER_COMPARE",
            turn,
            {
                "action": action,
                "same": active_courier == idle_courier,
                "active_c": active_courier,
                "idle_c": idle_courier,
                "active_a": boxes(active.frame(), 14),
                "active_empty": active_target["empty"],
                "idle_empty": idle_target["empty"],
            },
            flush=True,
        )


gkm_try.A.run_program("wa30", inspect)
