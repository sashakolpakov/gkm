"""Compact clean-room observation of the pristine bp35 level-9 entry."""

import json
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

import solve
from legs import COL_ANCHORS, ROW_ANCHORS, band_grid
from perception import color_counts, connected_components


def visible_components(frame):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=3)
        if blob.bbox[0] < 63 and blob.color != 10
    ]


def transition_counts(before, after):
    a = np.asarray(before)
    b = np.asarray(after)
    changed = (a != b)
    changed[63, :] = False
    pairs = Counter(
        (int(old), int(new))
        for old, new in zip(a[changed].tolist(), b[changed].tolist())
    )
    ys, xs = np.where(changed)
    bbox = None
    if len(ys):
        bbox = (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))
    return int(changed.sum()), bbox, sorted(pairs.items())


def observation(before, child):
    return {
        "level": int(child.levels_completed) + 1,
        "terminal": bool(child.terminal()),
        "delta": transition_counts(before, child.frame()),
        "grid": band_grid(child.frame()),
        "actors": [
            item for item in visible_components(child.frame()) if item[0] in (9, 11)
        ],
    }


def enter_level_9(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)
    solve.solve(env)


def probe(env):
    enter_level_9(env)
    frame = np.asarray(env.frame()).copy()
    print(
        "ENTRY",
        {
            "level": int(env.levels_completed) + 1,
            "terminal": bool(env.terminal()),
            "actions": env.actions,
            "colors": color_counts(frame),
        },
    )
    print("GRID")
    for row in band_grid(frame):
        print(row)
    print("COMPONENTS", visible_components(frame))

    for action in (3, 4, 7):
        child = env.clone()
        child.step(action)
        print("KEY", action, observation(frame, child))

    groups = defaultdict(list)
    representatives = {}
    for i, y in enumerate(ROW_ANCHORS):
        for j, x in enumerate(COL_ANCHORS):
            child = env.clone()
            child.step(6, x, y)
            key_frame = np.asarray(child.frame()).copy()
            key_frame[63, :] = 0
            key = (
                int(child.levels_completed),
                bool(child.terminal()),
                key_frame.tobytes(),
            )
            groups[key].append((i, j))
            representatives[key] = observation(frame, child)
    print("CLICK_GROUPS", len(groups))
    for cells, result in sorted(
        ((groups[key], representatives[key]) for key in groups),
        key=lambda item: (len(item[0]), item[0]),
    ):
        print("CLICK", cells, result)


arena.run_program("bp35", probe)
