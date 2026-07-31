import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players


def selected_cells(frame):
    grid = np.asarray(frame)
    points = []
    for row in range(0, 64, 3):
        for col in range(0, 64, 3):
            value = int(grid[row, col])
            if value in (8, 11):
                points.append((row // 3, col // 3, value))
            elif value == 0:
                r0, r1 = max(0, row - 1), min(64, row + 2)
                c0, c1 = max(0, col - 1), min(64, col + 2)
                if int((grid[r0:r1, c0:c1] == 0).sum()) >= 5:
                    points.append((row // 3, col // 3, value))
    r0 = min(row for row, _, _ in points)
    c0 = min(col for _, col, _ in points)
    return tuple(
        (row - r0, col - c0, value)
        for row, col, value in points
    )


def probe(env):
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    env.step(6, 28, 22)
    env.step(3)
    env.step(3)
    env.step(2)
    env.step(2)
    seen = {}
    for turns in range(21):
        signature = selected_cells(env.frame())
        shape = tuple((row, col) for row, col, _ in signature)
        green = tuple((row, col) for row, col, value in signature if value == 8)
        print(turns, "repeat", seen.get(signature), "shape", shape, "green", green)
        seen.setdefault(signature, turns)
        env.step(5)


arena.run_program("cn04", probe)
