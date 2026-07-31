"""Vary level 8's last manual target slot to shorten the courier tail."""

from itertools import product

import gkm_try

from probe8_entry import ReachedLevel8, StopAtLevel8
from probe8_tail_mutations import PREFIX, THIRD


def finish(start, actions, limit=17):
    clone = start.clone()
    base_level = clone.levels_completed
    used = 0
    for action in actions:
        if clone.terminal() or clone.levels_completed > base_level:
            break
        clone.step(action)
        used += 1
    while (
        used < limit
        and not clone.terminal()
        and clone.levels_completed == base_level
    ):
        clone.step(5)
        used += 1
    return clone.levels_completed > base_level, used


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel8(env))
    except ReachedLevel8:
        pass
    start = env.clone()
    for action in PREFIX + THIRD[:-3]:
        start.step(action)
    best = None
    clears = ((), (1,), (2,), (3,), (4,))
    for length in range(4):
        for moves in product((1, 2, 3, 4), repeat=length):
            for clear in clears:
                route = list(moves) + [5] + list(clear)
                won, used = finish(start, route)
                if won and (best is None or used < best[0]):
                    best = (used, route)
                    print("L8_DROP_BEST", best, flush=True)
    print("L8_DROP_RESULT", best, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
