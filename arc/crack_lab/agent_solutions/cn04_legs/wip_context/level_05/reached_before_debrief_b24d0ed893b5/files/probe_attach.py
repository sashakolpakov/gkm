import numpy as np

import gkm_try as harness
import players
from perception import color_counts


PATH = [
    [6, 36, 21], 5, 2, 2, 3, 3, 3, 3, 3, 3, 3,
    [6, 45, 42], 5, 5, 5, 1, 1, 1, 1,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    [6, 18, 48], 5, 1, 1, 1, 1, 1, 3, 3, 3,
]


def sig(env):
    frame = np.asarray(env.frame())
    counts = color_counts(frame)
    bg = max(counts, key=counts.get)
    return int(np.count_nonzero(frame[1:] != bg)), tuple(sorted(counts.items()))


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    players.play_level_3(env)
    for i, action in enumerate(PATH):
        before = sig(env)
        env.step(action)
        after = sig(env)
        print(i + 1, action, before, "=>", after, "level", env.levels_completed)
        if env.levels_completed > 3:
            break


harness.A.run_program("cn04", probe)
