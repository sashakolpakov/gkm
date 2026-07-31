"""Trace direct manual lower-depot deliveries after early dismissals."""

import gkm_try

from perception import arr
from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_reverse_idle import ROUTE
from probe8_reverse_stage import compact
from probe9_verify import boxes, tile_map

REVERSE_TOP = [4] * 3 + [1] * 3 + [5] * 3


def held(frame):
    grid = arr(frame)
    return int((grid == 0).sum())


def trace(env, actions, label):
    clone = env.clone()
    for turn, action in enumerate(actions, 1):
        clone.step(action)
        print(
            label,
            turn,
            action,
            boxes(clone.frame(), 14),
            held(clone.frame()),
            compact(clone, turn)["filled"],
            flush=True,
        )
    print(label + "_MAP", *tile_map(clone.frame()), sep="\n", flush=True)
    return clone


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass

    start = env.clone()
    for action in ROUTE:
        start.step(action)

    trace(start, [3, 3, 5] + [1] * 2 + [4] * 5 + [5], "DIRECT_A")
    trace(start, [3, 3, 5] + [1] * 2 + [4] * 6 + [5], "DIRECT_B")

    upper = env.clone()
    for action in REVERSE_TOP:
        upper.step(action)
    upper_delivery = [1, 1, 5] + [4] * 7 + [1, 5]
    trace(upper, upper_delivery, "UPPER_DELIVERY")
    upper_detour = [1, 1, 5, 2] + [4] * 7 + [1] * 2 + [5]
    trace(upper, upper_detour, "UPPER_DETOUR")


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
