import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np
from scipy import ndimage

import players


STAGES = (
    ("p0", [2] * 6 + [4] * 7),
    ("p1", [(6, 49, 7)] + [2] * 6),
    ("p4", [(6, 37, 46)] + [5] * 3),
    ("p3", [(6, 7, 40)] + [4] * 7),
    ("p2", [(6, 28, 22)] + [2] * 3 + [4] * 4 + [5] * 5),
)


def metric(frame):
    grid = np.asarray(frame)
    occupied = grid != 9
    return int(ndimage.label(occupied)[1]), int(occupied.sum())


def probe(env):
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    print("START", env.levels_completed, metric(env.frame()))
    for name, actions in STAGES:
        for action in actions:
            env.step(*action) if isinstance(action, tuple) else env.step(action)
            if env.levels_completed >= 6:
                print("FOUND", name, env.levels_completed)
                return
        print(name, env.levels_completed, metric(env.frame()))
    roots = {}
    frame = np.asarray(env.frame())
    for row in range(0, 64, 3):
        for col in range(0, 64, 3):
            if int(frame[row, col]) == 9:
                continue
            child = env.clone()
            child.step(6, col, row)
            roots.setdefault(np.asarray(child.frame()).tobytes(), ((col, row), child))
    print("SELECTIONS", len(roots))
    for click, child in roots.values():
        grid = np.asarray(child.frame())
        values, counts = np.unique(grid, return_counts=True)
        colors = {
            int(value): int(count)
            for value, count in zip(values, counts)
            if int(value) not in (4, 9)
        }
        marks = [
            (row // 3, col // 3, int(grid[row, col]))
            for row in range(0, 64, 3)
            for col in range(0, 64, 3)
            if int(grid[row, col]) in (0, 3, 8)
        ]
        print("SELECT", click, colors, marks)
    for click in ((0, 63), (63, 63), (18, 0), (54, 27)):
        child = env.clone()
        start = np.asarray(child.frame()).tobytes()
        changed_at = None
        for count in range(1, 11):
            child.step(6, *click)
            if np.asarray(child.frame()).tobytes() != start and changed_at is None:
                changed_at = count
            if child.levels_completed >= 6:
                print("WAIT_FOUND", click, count)
                return
        print("WAIT", click, "level", child.levels_completed, "changed", changed_at)
    for action in (1, 2, 3, 4, 5):
        child = env.clone()
        child.step(action)
        print("NEXT", action, child.levels_completed, metric(child.frame()))


arena.run_program("cn04", probe)
