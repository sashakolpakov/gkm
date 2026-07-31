import sys
from collections import Counter

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players


PATH = (3, 3, 3, 2, 2, 2, 2, 3, 3, 2, 2, 3, 3, 1, 1)


def tile(row, col):
    return slice(5 * row, 5 * row + 5), slice(5 * col - 1, 5 * col + 4)


def rows(frame, row_slice, col_slice):
    return tuple(
        "".join(f"{int(value):X}" for value in row)
        for row in np.asarray(frame)[row_slice, col_slice]
    )


def glyph(frame):
    return rows(frame, slice(55, 61), slice(3, 9))


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)
    print("target", rows(env.frame(), *tile(1, 2)))
    print("central_left", rows(env.frame(), *tile(6, 5)))
    print("central_right", rows(env.frame(), *tile(6, 7)))
    print("glyph0", glyph(env.frame()))
    for step, action in enumerate(PATH, 1):
        before = np.asarray(env.frame()).copy()
        env.step(action)
        after = np.asarray(env.frame())
        ys, xs = np.where(before[:60] != after[:60])
        groups = Counter((int(y // 5), int((x + 1) // 5)) for y, x in zip(ys, xs))
        histograms = []
        for pos, count in sorted(groups.items()):
            view = after[tile(*pos)]
            histograms.append((pos, count, sorted(Counter(map(int, view.flat)).items())))
        print(
            step,
            action,
            "groups",
            sorted(groups.items()),
            "after",
            histograms,
            "glyph",
            glyph(after),
        )


arena.run_program("ls20", probe)
