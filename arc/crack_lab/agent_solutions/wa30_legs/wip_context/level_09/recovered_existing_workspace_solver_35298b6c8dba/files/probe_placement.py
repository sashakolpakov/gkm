"""Verify placement and release semantics against the three-cell strip."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import ACTION_NAME, color_counts


SYMBOL = {0: "0", 1: "."}


def tile(frame, row, col):
    grid = np.asarray(frame)
    return "/".join(
        "".join(SYMBOL.get(int(v), f"{int(v):X}") for v in grid[r, col * 4 : col * 4 + 4])
        for r in range(row * 4, row * 4 + 4)
    )


def zone(frame):
    return {
        (row, col): tile(frame, row, col)
        for row in range(6, 11)
        for col in range(4, 12)
        if tile(frame, row, col) != "..../..../..../...."
    }


def probe(env):
    clone = env.clone()
    path = [1, 1, 5, 1, 1, 5]
    print("START", zone(clone.frame()))
    for action in path:
        clone.step(action)
        print(
            ACTION_NAME[action],
            "level",
            clone.levels_completed,
            "colors",
            color_counts(clone.frame()),
            "zone",
            zone(clone.frame()),
        )


arena.run_program("wa30", probe)
