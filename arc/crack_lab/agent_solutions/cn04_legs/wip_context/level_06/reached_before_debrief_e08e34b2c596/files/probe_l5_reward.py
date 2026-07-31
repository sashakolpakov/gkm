import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np
from scipy import ndimage

import players


PATH = (
    [5] * 4 + [1] * 3 + [4] * 6 + [(6, 54, 6)] + [5] * 3
    + [(6, 5, 38)] + [1] * 10 + [4] * 10
    + [(6, 47, 47)] + [5] * 3 + [1] * 10 + [4] * 2
)


def metric(frame):
    grid = np.asarray(frame)
    background = int(np.bincount(grid.ravel()).argmax())
    occupied = grid != background
    return (
        int(ndimage.label(occupied)[1]),
        int(occupied.sum()),
        int((grid == 8).sum()),
    )


def probe(env):
    while env.levels_completed < 4:
        getattr(players, f"play_level_{env.levels_completed + 1}")(env)
    previous = metric(env.frame())
    print("START", previous)
    for index, action in enumerate(PATH, 1):
        before_level = env.levels_completed
        before_frame = np.asarray(env.frame()).copy()
        env.step(*action) if isinstance(action, tuple) else env.step(action)
        current = metric(env.frame())
        if current != previous or isinstance(action, tuple) or env.levels_completed != before_level:
            print(
                index,
                action,
                "level",
                env.levels_completed,
                "before",
                metric(before_frame),
                "after",
                current,
            )
        previous = current


arena.run_program("cn04", probe)
