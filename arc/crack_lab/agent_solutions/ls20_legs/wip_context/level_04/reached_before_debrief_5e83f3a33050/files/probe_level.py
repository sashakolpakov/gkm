import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")

import gkm_arena as arena
import numpy as np

import players
from legs import DOWN, LEFT, RIGHT, UP


ROUTE = (
    (UP, 8),
    (LEFT, 1),
    (DOWN, 2),
    (RIGHT, 1),
    (LEFT, 1),
    (DOWN, 5),
    (LEFT, 1),
    (DOWN, 1),
    (RIGHT, 1),
    (UP, 3),
    (LEFT, 2),
    (UP, 1),
    (RIGHT, 7),
    (UP, 3),
    (LEFT, 1),
    (RIGHT, 1),
    (LEFT, 1),
    (RIGHT, 1),
    (UP, 1),
    (DOWN, 1),
)


def hud_glyph(frame):
    return np.asarray(frame)[55:61:2, 3:9:2].copy()


def symbolic_glyph(glyph):
    return tuple(
        "".join("." if int(value) == 5 else f"{int(value):X}" for value in row)
        for row in glyph
    )


def probe(env):
    players.play_level_1(env)
    players.play_level_2(env)
    clone = env.clone()
    target = np.asarray(clone.frame())[51:54, 55:58].copy()
    print("entry", int(clone.levels_completed), symbolic_glyph(hud_glyph(clone.frame())))

    step = 0
    for action, count in ROUTE:
        for _ in range(count):
            clone.step(action)
            step += 1
            if step in (21, 38, 40, 43):
                glyph = hud_glyph(clone.frame())
                mismatch = int(np.count_nonzero(glyph != target))
                print(
                    "probe",
                    step,
                    int(clone.levels_completed),
                    symbolic_glyph(glyph),
                    "mismatch",
                    mismatch,
                )


arena.run_program("ls20", probe)
