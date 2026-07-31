"""Trace target progress after the verified early level-8 dismissals."""

import gkm_try

from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_trace import target_state
from probe9_verify import boxes


ROUTE = (
    [4] * 3 + [1] * 3 + [5] * 3
    + [2] * 3 + [4] * 5 + [2] * 5
    + [2, 2, 3, 3, 1, 5]
)


def compact(env, turn):
    empty, filled = target_state(env.frame())
    return {
        "turn": turn,
        "level": env.levels_completed,
        "terminal": env.terminal(),
        "avatar": boxes(env.frame(), 14),
        "helpers": boxes(env.frame(), 12),
        "competitors": boxes(env.frame(), 15),
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
    print("REVERSE_START", compact(clone, len(ROUTE)), flush=True)

    prior = target_state(clone.frame())
    for turn in range(len(ROUTE) + 1, 91):
        clone.step(5)
        current = target_state(clone.frame())
        if current != prior or clone.levels_completed > base_level:
            print("REVERSE_EVENT", compact(clone, turn), flush=True)
        prior = current
        if clone.terminal() or clone.levels_completed > base_level:
            break


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
