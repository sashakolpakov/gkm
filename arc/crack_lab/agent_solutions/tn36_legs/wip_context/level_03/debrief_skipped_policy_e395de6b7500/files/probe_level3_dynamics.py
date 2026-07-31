"""Bounded contextual probes for level-3 controls and board state."""

import json
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as A
import numpy as np

from perception import connected_components


BUTTONS = {
    "D": (6, 5, 58),
    "U": (6, 15, 58),
    "L": (6, 25, 58),
    "R": (6, 35, 58),
    "S": (6, 57, 58),
}
SEGMENT_ROWS = (33, 36, 39, 42, 45, 48)
LEFT_COLUMNS = (8, 13, 18, 23)
RIGHT_COLUMNS = (34, 39, 44, 49, 54, 59)


def region_blob(frame, color, r0, c0, r1, c1):
    points = np.argwhere(frame[r0:r1, c0:c1] == color)
    if not len(points):
        return None
    points += (r0, c0)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    mean = points.mean(axis=0)
    return {
        "count": len(points),
        "bbox": tuple(int(value) for value in (*lo, *hi)),
        "centroid": tuple(round(float(value), 2) for value in mean),
        "mask": tuple(
            "".join("#" if frame[row, col] == color else "." for col in range(lo[1], hi[1] + 1))
            for row in range(lo[0], hi[0] + 1)
        ),
    }


def panel(frame, columns):
    return tuple("".join(str(int(frame[row, col])) for col in columns) for row in SEGMENT_ROWS)


def state(env):
    frame = np.asarray(env.frame())
    agents = [
        (blob.bbox, blob.area, tuple(round(value, 2) for value in blob.centroid))
        for blob in connected_components(frame, colors=(11,), min_area=4)
        if blob.bbox[0] < 32
    ]
    return {
        "levels": env.levels_completed,
        "cursor": region_blob(frame, 4, 3, 1, 32, 31),
        "agents": agents,
        "left": panel(frame, LEFT_COLUMNS),
        "right": panel(frame, RIGHT_COLUMNS),
        "selected": "".join(str(int(frame[54, col])) for col in (5, 15, 25, 35)),
        "timer": int(np.count_nonzero(frame[1] == 9)),
    }


def observe(env):
    with open("checkpoint.json") as stream:
        checkpoint = json.load(stream)
    for action in checkpoint["final_path"]:
        env.step(action)

    experiments = (
        "",
        "D",
        "U",
        "L",
        "R",
        "UU",
        "DD",
        "UR",
        "UD",
        "RL",
        "ULDR",
        "URDL",
        "LURD",
    )
    for experiment in experiments:
        clone = env.clone()
        for command in experiment:
            clone.step(*BUTTONS[command])
        print(experiment or "BASE", state(clone))


levels, path, error = A.run_program("tn36", observe)
print("probe_result", {"levels": levels, "moves": len(path), "error": error})
