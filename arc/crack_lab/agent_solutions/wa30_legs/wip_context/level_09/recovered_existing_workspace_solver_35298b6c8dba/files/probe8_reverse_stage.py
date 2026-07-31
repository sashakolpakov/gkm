"""Test a compact bottom-port stage after the early level-8 dismissals."""

import gkm_try

from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_reverse_idle import ROUTE
from probe8_trace import target_state
from probe9_verify import boxes


STAGE_LOWER = (
    [3, 3, 5]
    + [1] * 2 + [4] * 4 + [5, 2]
)


def compact(env, turn):
    empty, filled = target_state(env.frame())
    return {
        "turn": turn,
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "avatar": boxes(env.frame(), 14),
        "helpers": boxes(env.frame(), 12),
        "cargo": boxes(env.frame(), 4),
        "empty": empty,
        "filled": filled,
    }


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass

    base_level = env.levels_completed
    clone = env.clone()
    for action in ROUTE:
        clone.step(action)
    for offset, action in enumerate(STAGE_LOWER, 1):
        clone.step(action)
        if action == 5:
            print(
                "STAGE_USE",
                compact(clone, len(ROUTE) + offset),
                flush=True,
            )
    print(
        "STAGE_LOWER_DONE",
        compact(clone, len(ROUTE) + len(STAGE_LOWER)),
        flush=True,
    )

    prior = target_state(clone.frame())
    start = len(ROUTE) + len(STAGE_LOWER)
    for turn in range(start + 1, 131):
        clone.step(5)
        current = target_state(clone.frame())
        if current != prior or clone.levels_completed > base_level:
            print("STAGE_EVENT", compact(clone, turn), flush=True)
        prior = current
        if clone.terminal() or clone.levels_completed > base_level:
            break


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
