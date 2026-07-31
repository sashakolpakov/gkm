"""Reward probes for the initial 3x5 toggle field."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS


def clicks(cells):
    return [(6, COL_ANCHORS[column], ROW_ANCHORS[row]) for row, column in cells]


UPPER = [(row, column) for row in range(5) for column in range(2, 5)]
PATTERNS = {
    "all_upper": UPPER,
    "all_cells": UPPER + [(6, 4), (8, 4)],
    "left_column": [(row, 2) for row in range(5)],
    "middle_column": [(row, 3) for row in range(5)],
    "right_column": [(row, 4) for row in range(5)],
    "top_row": [(0, column) for column in range(2, 5)],
    "bottom_row": [(4, column) for column in range(2, 5)],
    "down_diagonal": [(0, 2), (1, 3), (2, 4)],
    "up_diagonal": [(2, 2), (1, 3), (0, 4)],
    "x_shape": [(0, 2), (0, 4), (1, 3), (2, 2), (2, 4)],
}


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    base_level = int(env.levels_completed)
    for name, cells in PATTERNS.items():
        clone = env.clone()
        for action in clicks(cells):
            clone.step(*action)
            if clone.terminal() or clone.levels_completed > base_level:
                break
        print(
            name,
            {
                "steps": len(cells),
                "level_delta": int(clone.levels_completed) - base_level,
                "terminal": bool(clone.terminal()),
            },
        )


arena.run_program("bp35", probe)
