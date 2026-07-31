"""Enumerate achieved level-6 top-left positions for each selected piece."""
from collections import deque
import sys
import time

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


def bbox(env):
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
    started = time.monotonic()
    steps = 0
    maps = {}
    for piece, point in POINTS.items():
        root = env.clone()
        root.step(6, *point)
        start = bbox(root)
        seen = {start}
        queue = deque([root])
        while queue and len(seen) < 500:
            node = queue.popleft()
            for action in (1, 2, 3, 4):
                child = node.clone()
                try:
                    child.step(action)
                except IndexError:
                    continue
                steps += 1
                key = bbox(child)
                if key in seen:
                    continue
                seen.add(key)
                queue.append(child)
                delay = steps / 280.0 - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)
        by_top = {}
        for top, left, _, _ in sorted(seen):
            by_top.setdefault(top, []).append(left)
        maps[piece] = {
            top: tuple(lefts) for top, lefts in by_top.items()
        }
    print("L6_REACH2D", maps, "STEPS", steps)


if __name__ == "__main__":
    levels, path, err = Arena.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
