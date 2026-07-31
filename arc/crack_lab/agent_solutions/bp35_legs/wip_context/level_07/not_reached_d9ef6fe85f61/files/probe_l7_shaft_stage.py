"""Find initial supports that make the direct right-shaft descent survivable."""

from itertools import combinations
import json
import os
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from legs import COL_ANCHORS, ROW_ANCHORS, run_actions
from probe_l7_frontier import R, avatar, summary


UPPER = [(row, column) for row in range(5) for column in range(2, 5)]


def action(cell):
    row, column = cell
    return 6, COL_ANCHORS[column], ROW_ANCHORS[row]


DESCEND = [
    R,
    R,
    R,
    action((8, 4)),
    (6, 3, 3),
    R,
    R,
]


def probe(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for step in prefix:
        env.step(step)
    root = env.clone()
    base_level = int(env.levels_completed)
    survivors = []
    min_size = int(os.environ.get("MIN_STAGE", "0"))
    max_size = int(os.environ.get("MAX_STAGE", "2"))
    for size in range(min_size, max_size + 1):
        for staged in combinations(UPPER, size):
            node = root.clone()
            run_actions(node, [action(cell) for cell in staged] + DESCEND)
            if node.terminal() or avatar(node.frame()) is None:
                continue
            state = summary(node)
            survivors.append(
                {
                    "staged": staged,
                    "level_delta": int(node.levels_completed) - base_level,
                    "avatar": state["avatar"],
                    "grid": state["grid"],
                }
            )
    print(
        "RESULT",
        {
            "tested_through": max_size,
            "survivors": len(survivors),
            "items": survivors,
        },
    )


arena.run_program("bp35", probe)
