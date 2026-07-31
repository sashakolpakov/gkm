"""Enumerate contextual actions after the first safe gravity round trip."""

import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import COL_ANCHORS, ROW_ANCHORS, band_grid, moves_used
from perception import connected_components


FIRST_STAGE = [(6, 3, 39), 4, 4, 4, (6, 3, 5)]


def enter_stage(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)
    for action in FIRST_STAGE:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def compact(env):
    blobs = connected_components(env.frame(), min_area=3)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
        "grid": tuple("".join(row) for row in band_grid(env.frame())),
        "pieces": tuple(
            (b.color, b.bbox, b.area)
            for b in blobs
            if b.bbox[0] < 63 and b.color not in (5, 10)
        ),
    }


def signature(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return int(env.levels_completed), bool(env.terminal()), frame.tobytes()


def step(root, action):
    child = root.clone()
    child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def probe(env):
    enter_stage(env)
    print("STAGE", compact(env))
    for action in (3, 4, 7):
        print("KEY", action, compact(step(env, action)))

    groups = defaultdict(list)
    representatives = {}
    click_points = [
        (6, x, y)
        for y in ROW_ANCHORS
        for x in COL_ANCHORS
    ]
    for action in click_points:
        child = step(env, action)
        key = signature(child)
        groups[key].append((action[2] // 6, (action[1] - 15) // 6))
        representatives[key] = compact(child)
    for cells, result in sorted(
        ((groups[key], representatives[key]) for key in groups),
        key=lambda item: (len(item[0]), item[0]),
    ):
        print("CLICK", cells, result)


arena.run_program("bp35", probe)
