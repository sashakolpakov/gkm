"""Compact, read-only level-7 probes through the documented arena surface."""

import json
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import color_counts, connected_components, frame_delta
from probe_l7_raw_search import target_path_distance


ROWS = [3 + 6 * i for i in range(10)]
COLS = [15 + 6 * j for j in range(8)]


def enter_level_7(env):
    with open("checkpoint.json") as stream:
        prefix = json.load(stream)["final_path"]
    for action in prefix:
        env.step(action)


def components(frame, min_area=3):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=min_area)
    ]


def token_cells(frame, colors):
    grid = np.asarray(frame)
    return {
        color: [
            (i, j)
            for i, row in enumerate(ROWS)
            for j, col in enumerate(COLS)
            if int(grid[row, col]) == color
        ]
        for color in colors
    }


def component_changes(before, after):
    old = set(components(before))
    new = set(components(after))
    return {"gone": sorted(old - new), "new": sorted(new - old)}


def compact_delta(before, after):
    delta = frame_delta(before, after)
    return {"count": delta["count"], "bbox": delta["bbox"]}


def probe(env):
    enter_level_7(env)
    base = np.asarray(env.frame()).copy()
    print(
        "ENTRY",
        {
            "level": int(env.levels_completed) + 1,
            "actions": list(env.actions),
            "terminal": bool(env.terminal()),
        },
    )
    print("OBJECTS", components(base))
    print("COLORS", color_counts(base))
    print("TOKENS", token_cells(base, (9, 12, 14, 15)))
    gravity = env.clone()
    gravity.step(6, 3, 3)
    print("TARGET_PATH_DISTANCE", target_path_distance(gravity.frame()))
    for movement in (3, 4):
        moved = gravity.clone()
        moved.step(movement)
        print("TARGET_PATH_AFTER", movement, target_path_distance(moved.frame()))
    print("LATTICE", [[int(base[row, col]) for col in COLS] for row in ROWS])
    for action in (3, 4):
        clone = env.clone()
        clone.step(action)
        after = clone.frame()
        print(
            "ACTION",
            action,
            {
                "level_delta": int(clone.levels_completed - env.levels_completed),
                "terminal": bool(clone.terminal()),
                "delta": compact_delta(base, after),
                "objects": component_changes(base, after),
            },
        )
    for action_kind in (6, 7):
        for color in (5, 10, 12, 15, 9):
            target = next(
                (
                    (col, row)
                    for row in ROWS
                    for col in COLS
                    if int(base[row, col]) == color
                ),
                None,
            )
            if target is None:
                continue
            clone = env.clone()
            try:
                clone.step(action_kind, *target)
            except Exception as error:
                print("COORD_ERROR", action_kind, type(error).__name__, str(error))
                break
            after = clone.frame()
            print(
                "COORD",
                action_kind,
                color,
                target,
                {
                    "level_delta": int(clone.levels_completed - env.levels_completed),
                    "terminal": bool(clone.terminal()),
                    "delta": compact_delta(base, after),
                    "objects": component_changes(base, after),
                },
            )


arena.run_program("bp35", probe)
