"""Enumerate actions at the first staged downward landing."""

import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import COL_ANCHORS, ROW_ANCHORS, band_grid, click_action, moves_used
from perception import color_counts, connected_components


def enter_down_stage(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)
    route = [
        (6, 3, 39), 4, 4, 4, (6, 3, 5),
        click_action(5, 4),
        click_action(1, 2),
        click_action(6, 3), 3,
        click_action(6, 2), 3,
        *([click_action(5, 2)] * 5),
        (6, 3, 9),
    ]
    for action in route:
        env.step(*action) if isinstance(action, tuple) else env.step(action)


def compact(env):
    blobs = connected_components(env.frame(), colors=(7, 8, 9, 14, 15), min_area=3)
    return {
        "level": int(env.levels_completed) + 1,
        "terminal": bool(env.terminal()),
        "moves": moves_used(env.frame()),
        "colors": color_counts(env.frame()),
        "grid": tuple("".join(row) for row in band_grid(env.frame())),
        "pieces": tuple(
            (b.color, b.bbox, b.area) for b in blobs if b.bbox[0] < 63
        ),
    }


def signature(env):
    frame = np.asarray(env.frame()).copy()
    frame[63, :] = 0
    return int(env.levels_completed), bool(env.terminal()), frame.tobytes()


def stepped(root, action):
    child = root.clone()
    child.step(*action) if isinstance(action, tuple) else child.step(action)
    return child


def probe(env):
    enter_down_stage(env)
    print("DOWN", compact(env))
    for action in (3, 4, 7):
        print("KEY", action, compact(stepped(env, action)))

    groups = defaultdict(list)
    representatives = {}
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            action = (6, x, y)
            child = stepped(env, action)
            key = signature(child)
            groups[key].append((i, j))
            representatives[key] = compact(child)
    for cells, result in sorted(
        ((groups[key], representatives[key]) for key in groups),
        key=lambda item: (len(item[0]), item[0]),
    ):
        print("CLICK", cells, result)

    child = env.clone()
    for index in range(1, 7):
        if child.terminal():
            break
        child.step(4)
        print("RIGHT", index, compact(child))


arena.run_program("bp35", probe)
