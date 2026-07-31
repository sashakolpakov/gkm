"""Vary level 3's final courier-port drop to shorten its delivery tail."""

import gkm_try


class ReachedLevel3(Exception):
    pass


class StopAtLevel3:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action, *args):
        result = self.env.step(action, *args)
        if self.env.levels_completed >= 2:
            raise ReachedLevel3
        return result


PREFIX = (
    [3] * 2 + [2] * 2 + [4, 5] + [1] * 3 + [4] * 14 + [5, 3]
    + [3] * 2 + [1] * 3 + [4, 5] + [1] * 2 + [4] * 3 + [5, 3]
    + [3] * 5 + [2, 4, 5] + [2] * 4
)


def finish(start, actions, limit=28):
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
        gkm_try.resumed_solve(StopAtLevel3(env))
    except ReachedLevel3:
        pass
    start = env.clone()
    for action in PREFIX:
        start.step(action)
    best = None
    offsets = ([], [1], [1, 1], [2], [2, 2])
    clears = ([], [1], [2], [3], [4])
    for right_steps in range(3, 10):
        for before in offsets:
            for after in offsets:
                for clear in clears:
                    candidates = (
                        before + [4] * right_steps + after + [5] + clear,
                        [4] * right_steps + before + after + [5] + clear,
                    )
                    for route in candidates:
                        won, used = finish(start, route)
                        if won and (best is None or used < best[0]):
                            best = (used, route)
                            print("L3_DROP_BEST", best, flush=True)
    print("L3_DROP_RESULT", best, flush=True)


if __name__ == "__main__":
    gkm_try.A.run_program("wa30", inspect)
