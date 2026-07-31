"""Try nearby placements of level 5's final held cargo."""

from itertools import product

import gkm_try

from probe5_structure import PHASES
from probe5_tail_bfs import ReachedLevel5, StopAtLevel5


def finish(start, actions, limit=13):
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
        gkm_try.resumed_solve(StopAtLevel5(env))
    except ReachedLevel5:
        pass
    start = env.clone()
    prefix = (
        PHASES[0][1] + PHASES[1][1] + PHASES[2][1]
        + PHASES[3][1][:-3]
    )
    for action in prefix:
        start.step(action)
    best = None
    clears = ((), (1,), (2,), (3,), (4,))
    for length in (4,):
        for moves in product((1, 2, 3, 4), repeat=length):
            for clear in clears:
                route = list(moves) + [5] + list(clear)
                won, used = finish(start, route)
                if won and (best is None or used < best[0]):
                    best = (used, route)
                    print("L5_VARIANT_BEST", best, flush=True)
    print("L5_VARIANT_RESULT", best, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
