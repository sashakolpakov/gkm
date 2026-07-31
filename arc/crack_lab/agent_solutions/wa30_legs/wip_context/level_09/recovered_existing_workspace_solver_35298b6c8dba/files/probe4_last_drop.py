"""Vary level 4's final courier-port release."""

from itertools import product

import gkm_try


class ReachedLevel4(Exception):
    pass


class StopAtLevel4:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        result = self.env.step(action, *args)
        if self.env.levels_completed >= 3:
            raise ReachedLevel4
        return result


RUNS = (
    (4, 1), (2, 1), (5, 1), (2, 2), (5, 1),
    (4, 1), (5, 1), (4, 1), (5, 1), (3, 2), (5, 1),
    (1, 2), (3, 1), (5, 1), (4, 1), (1, 2), (3, 1), (5, 1),
    (2, 1), (3, 1), (5, 1), (4, 2), (1, 1), (5, 1),
    (2, 2), (5, 1), (3, 1), (1, 2), (4, 1), (2, 1), (5, 1),
    (2, 3), (5, 1), (1, 4), (4, 1), (5, 1), (4, 1), (5, 1),
    (3, 1), (5, 9),
)
ROUTE = [action for action, count in RUNS for _ in range(count)]
PREFIX = ROUTE[:-11]


def finish(start, actions, limit=10):
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
        gkm_try.resumed_solve(StopAtLevel4(env))
    except ReachedLevel4:
        pass
    start = env.clone()
    for action in PREFIX:
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
                    print("L4_DROP_BEST", best, flush=True)
    print("L4_DROP_RESULT", best, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
