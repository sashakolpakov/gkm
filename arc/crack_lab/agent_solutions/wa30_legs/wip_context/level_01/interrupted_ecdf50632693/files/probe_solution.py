"""Verify a dense-progress plan entirely on a pristine clone."""

import itertools
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as arena

from perception import ACTION_NAME, connected_components


TARGETS = ((7, 7), (7, 8), (7, 9))


def player_cell(frame):
    orange = connected_components(frame, colors=[14], min_area=1)
    blob = orange[0]
    return (round(blob.centroid[0] / 4), round(blob.centroid[1] / 4))


def box_cells(frame):
    grid = np.asarray(frame)
    out = []
    for row in range(15):
        for col in range(16):
            tile = grid[row * 4 : row * 4 + 4, col * 4 : col * 4 + 4]
            rim = np.concatenate((tile[0], tile[3], tile[1:3, 0], tile[1:3, 3]))
            if (
                len(tile) == 4
                and tile.shape[1] == 4
                and np.all(tile[1:3, 1:3] == 9)
                and len(np.unique(rim)) == 1
                and int(rim[0]) in (0, 3, 4)
            ):
                out.append((row, col))
    return tuple(out)


def dense(frame):
    boxes = box_cells(frame)
    on_target = sum(box in TARGETS for box in boxes)
    distance = min(
        sum(abs(box[0] - target[0]) + abs(box[1] - target[1])
            for box, target in zip(boxes, ordering))
        for ordering in itertools.permutations(TARGETS)
    )
    return boxes, on_target, distance


SEGMENTS = (
    ("C_attach", [1, 1, 5]),
    ("C_place", [1, 1, 5]),
    ("reach_B", [3, 3, 3, 3, 3, 1, 4, 5]),
    ("B_place", [4, 4, 4, 5]),
    ("reach_A", [1, 1, 4, 4, 4, 4, 4, 4, 2, 3, 5]),
    ("A_place", [2, 3, 3, 5]),
)


def probe(env):
    clone = env.clone()
    base_level = int(clone.levels_completed)
    steps = 0
    print("START", player_cell(clone.frame()), dense(clone.frame()))
    for name, path in SEGMENTS:
        for action in path:
            clone.step(action)
            steps += 1
            print(
                steps,
                ACTION_NAME[action],
                player_cell(clone.frame()),
                dense(clone.frame()),
                "level",
                clone.levels_completed,
            )
            if clone.levels_completed > base_level:
                print("SOLVED", steps, name)
                return
        print("MILESTONE", name)
    print("UNSOLVED", clone.levels_completed)


arena.run_program("wa30", probe)
