import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players
from perception import color_counts, connected_components


CANDIDATE = (
    [4] * 4
    + [(6, 49, 7)] + [2] + [3] * 10
    + [(6, 28, 22)] + [2] * 2 + [5] + [1] * 8
    + [(6, 7, 40)] + [4] * 6 + [5] + [1] * 13 + [3] * 3
    + [(6, 37, 46)] + [1] * 4 + [5] + [1] * 7 + [3] * 9
)


def probe(env):
    while env.levels_completed < 5:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    for index, action in enumerate(CANDIDATE, 1):
        before = env.levels_completed
        env.step(*action) if isinstance(action, tuple) else env.step(action)
        if isinstance(action, tuple) or env.levels_completed != before:
            print(index, action, "level", env.levels_completed,
                  "colors", color_counts(env.frame()))
        if env.levels_completed >= 6:
            print("FOUND", CANDIDATE[:index])
            return
    print("END", env.levels_completed, color_counts(env.frame()))
    roots = {}
    frame = np.asarray(env.frame())
    background = max(color_counts(frame), key=color_counts(frame).get)
    for row in range(1, 64, 3):
        for col in range(1, 64, 3):
            if int(frame[row, col]) == background:
                continue
            child = env.clone()
            child.step(6, col, row)
            roots.setdefault(np.asarray(child.frame()).tobytes(), ((col, row), child))
    print("SELECTIONS", len(roots))
    for click, child in roots.values():
        marks = [
            (blob.color, blob.bbox, blob.area)
            for blob in connected_components(child.frame(), colors=(3, 8), min_area=4)
        ]
        print(" SELECT", click, color_counts(child.frame()), marks)
    for action in range(1, 6):
        child = env.clone()
        child.step(action)
        print("NEXT", action, child.levels_completed, color_counts(child.frame()))


arena.run_program("cn04", probe)
