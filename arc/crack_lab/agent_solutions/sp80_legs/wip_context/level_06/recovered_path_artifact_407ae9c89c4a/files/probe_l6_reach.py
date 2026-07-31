"""Observed one-axis reachability for each level-6 movable piece."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as Arena

from players import (
    play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
)


POINTS = {
    "A": (30, 19),
    "B": (45, 18),
    "C": (25, 33),
    "D": (31, 46),
}


def selected_bbox(env):
    pixels = np.asarray(env.frame())
    rows, cols = np.where(pixels == 9)
    return (
        int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()),
    )


def probe(env):
    for player in (
        play_level_1, play_level_2, play_level_3, play_level_4, play_level_5,
    ):
        player(env)
    results = {}
    for piece, point in POINTS.items():
        for direction in (1, 2, 3, 4):
            node = env.clone()
            node.step(6, *point)
            boxes = [selected_bbox(node)]
            for _ in range(20):
                node.step(direction)
                box = selected_bbox(node)
                if box == boxes[-1]:
                    break
                boxes.append(box)
            results[(piece, direction)] = tuple(boxes)
    print("L6_REACH", results)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
