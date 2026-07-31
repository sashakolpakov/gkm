import numpy as np

import gkm_try as H

from perception import connected_components
from probe_targets8 import center, groups


PATH = (
    (6, 7, 48),
    (6, 7, 54),
    (6, 15, 19),
    (6, 7, 19),
    (6, 16, 53),
    (6, 31, 44),
    (6, 42, 43),
    (6, 10, 53),
    (6, 7, 55),
    (6, 37, 49),
    (6, 8, 19),
    (6, 8, 31),
    (6, 10, 42),
    (6, 20, 43),
    (6, 21, 43),
    (6, 30, 50),
    (6, 39, 49),
    (6, 40, 51),
    (6, 41, 61),
    (6, 36, 55),
    (6, 46, 55),
    (6, 56, 53),
)


def inspect(env):
    H.resumed_solve(env)
    start = int(env.levels_completed)
    initial = env.frame()
    ring_masks = tuple(
        frozenset(
            (row, col)
            for row in range(blob.bbox[0], blob.bbox[2] + 1)
            for col in range(blob.bbox[1], blob.bbox[3] + 1)
            if int(initial[row][col]) == 9
        )
        for blob in connected_components(initial, colors=(9,), min_area=9)
        if blob.bbox[0] >= 10
    )

    def summary():
        frame = np.asarray(env.frame())
        counts = tuple(
            (color, int(np.count_nonzero(frame[10:] == color)))
            for color in (7, 8, 11, 12, 14)
        )
        if int(env.levels_completed) != start:
            return counts
        items = tuple(
            frozenset(
                (row, col)
                for row in range(blob.bbox[0], blob.bbox[2] + 1)
                for col in range(blob.bbox[1], blob.bbox[3] + 1)
            )
            for blob in connected_components(frame, colors=(12,), min_area=25)
            if blob.bbox[0] >= 10
        ) + tuple(
            frozenset(group)
            for color in (7, 14)
            for group in groups(frame, color)
        )
        occupancy = tuple(
            max((len(item & mask) for item in items), default=0)
            for mask in ring_masks
        )
        agents = tuple(
            (color, center(group))
            for color in (7, 14)
            for group in groups(frame, color)
        )
        return counts, agents, occupancy

    print("START", start, summary())
    for index, action in enumerate(PATH, 1):
        env.step(*action)
        if index in (1, 4, 7, 9, 11, 18, 19, 20, 21, len(PATH)):
            print(
                "STEP", index, action,
                "level", int(env.levels_completed),
                "terminal", bool(env.terminal()),
                summary(),
            )
    print("DELTA", int(env.levels_completed) - start)


if __name__ == "__main__":
    H.A.run_program("su15", inspect)
