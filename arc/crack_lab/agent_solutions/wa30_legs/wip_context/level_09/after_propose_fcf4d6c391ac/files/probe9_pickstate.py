"""Compare the local-cargo contact and pickup observational states."""

import gkm_try

from perception import arr
from probe9_candidate import direct_second_prefix
from probe9_verify import ReachedLevel9, StopAtLevel9, boxes


def describe(env):
    grid = arr(env.frame())
    return {
        "avatar": boxes(env.frame(), 14),
        "cargo": boxes(env.frame(), 4),
        "local_cells": {
            (row, col): tuple(sorted(set(int(value) for value in
                grid[row * 4:row * 4 + 4,
                     col * 4:col * 4 + 4].flat)))
            for row in (7, 8)
            for col in (1, 2)
        },
    }


def inspect(env):
    try:
        gkm_try.resumed_solve(StopAtLevel9(env))
    except ReachedLevel9:
        pass

    start = env.clone()
    for action in direct_second_prefix():
        start.step(action)
    dismiss = [2, 2, 3, 5]
    paths = {
        "beside": dismiss + [3] * 3,
        "contact": dismiss + [3] * 4,
        "shortcut_use": dismiss + [3] * 3 + [5],
        "verified_pick": dismiss + [3] * 4 + [5],
    }
    for name, path in paths.items():
        state = start.clone()
        for action in path:
            state.step(action)
        print("PICKSTATE", name, len(path) + 40, describe(state),
              flush=True)


gkm_try.A.run_program("wa30", inspect)
