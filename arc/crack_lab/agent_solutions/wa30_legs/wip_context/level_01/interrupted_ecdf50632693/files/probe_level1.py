"""Compact clean-room observations for wa30 level 1."""

import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import ACTION_NAME, action_deltas, color_counts, connected_components


SYMBOL = {1: "1", 2: "2", 4: "4", 7: "7", 9: "9", 14: "E"}


def macro(frame, cell=4):
    grid = np.asarray(frame)
    rows = []
    for r in range(0, grid.shape[0], cell):
        row = []
        for c in range(0, grid.shape[1], cell):
            tile = grid[r : r + cell, c : c + cell]
            values, counts = np.unique(tile, return_counts=True)
            color = int(values[np.argmax(counts)])
            row.append(SYMBOL.get(color, "?"))
        rows.append("".join(row))
    return rows


def changed_macro(before, after, cell=4):
    a, b = np.asarray(before), np.asarray(after)
    out = []
    for r in range(0, a.shape[0], cell):
        for c in range(0, a.shape[1], cell):
            if np.any(a[r : r + cell, c : c + cell] != b[r : r + cell, c : c + cell]):
                out.append((r // cell, c // cell))
    return out


def compact_blobs(frame):
    return [
        (blob.color, blob.bbox, blob.area)
        for blob in connected_components(frame, min_area=4)
    ]


def probe(env):
    frame = env.frame()
    print("ACTIONS", env.actions)
    print("LEVEL", env.levels_completed, "TERMINAL", env.terminal())
    print("COLORS", color_counts(frame))
    print("MACRO")
    print("\n".join(macro(frame)))
    print("BLOBS", compact_blobs(frame))
    deltas = action_deltas(env, env.actions)
    for action in env.actions:
        clone = env.clone()
        clone.step(action)
        print(
            "ACTION",
            action,
            ACTION_NAME.get(action, str(action)),
            "DELTA",
            deltas[action]["count"],
            deltas[action]["bbox"],
            "CELLS",
            changed_macro(frame, clone.frame()),
            "LEVEL",
            clone.levels_completed,
        )


arena.run_program("wa30", probe)
