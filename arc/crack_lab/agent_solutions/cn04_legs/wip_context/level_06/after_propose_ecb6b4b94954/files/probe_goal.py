import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import numpy as np
import gkm_arena as arena
from scipy import ndimage


PATH = [2] * 7 + [4] * 4 + [5] * 3


def summary(frame):
    array = np.asarray(frame)
    background = int(np.bincount(array.ravel()).argmax())
    labels, count = ndimage.label(array != background)
    sizes = sorted((int((labels == i).sum()) for i in range(1, count + 1)),
                   reverse=True)
    colors, amounts = np.unique(array, return_counts=True)
    return int(count), sizes, dict(zip(map(int, colors), map(int, amounts)))


def probe(env):
    print("START", env.levels_completed, summary(env.frame()))
    for index, action in enumerate(PATH, 1):
        before = env.levels_completed
        env.step(action)
        if index >= len(PATH) - 3 or env.levels_completed != before:
            print(index, action, "level", env.levels_completed,
                  summary(env.frame()))


arena.run_program("cn04", probe)
