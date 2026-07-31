import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players


SETUPS = (
    (None, [2] * 5 + [4] * 10),
    ((49, 7), [2] * 5 + [3] * 4),
    ((28, 22), [4] * 5),
    ((7, 40), [1] * 6 + [4] * 10),
    ((37, 46), [1] * 8 + [4] * 2),
)


def cells(frame):
    grid = np.asarray(frame)
    selected = []
    for row in range(0, 64, 3):
        for col in range(0, 64, 3):
            value = int(grid[row, col])
            if value not in (4, 9):
                selected.append((row // 3, col // 3, value))
    if not selected:
        return ()
    r0 = min(row for row, _, _ in selected)
    c0 = min(col for _, col, _ in selected)
    return tuple((row - r0, col - c0, value) for row, col, value in selected)


def compact(points):
    return [
        (row, col, "G" if value == 8 else "X" if value == 0 else "B")
        for row, col, value in points
    ]


def probe(env):
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    for click, setup in SETUPS:
        node = env.clone()
        if click is not None:
            node.step(6, *click)
        for action in setup:
            node.step(action)
        print("PIECE", click)
        seen = {}
        for turns in range(5):
            signature = cells(node.frame())
            shape = tuple((row, col) for row, col, _ in signature)
            visible = [
                (row, col)
                for row in range(0, 64, 3)
                for col in range(0, 64, 3)
                if int(np.asarray(node.frame())[row, col]) not in (0, 4, 9)
            ]
            bounds = (
                min(row for row, _ in visible) // 3,
                min(col for _, col in visible) // 3,
                max(row for row, _ in visible) // 3,
                max(col for _, col in visible) // 3,
            )
            print(
                turns,
                "repeat",
                seen.get(shape),
                "bounds",
                bounds,
                compact(signature),
            )
            seen.setdefault(shape, turns)
            node.step(5)


arena.run_program("cn04", probe)
