"""Tiny pixel crops for identifying the distinct level-5 objects."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from probe_level5 import reach_level_5


OBJECTS = {
    "avatar": (40, 45, 49, 54),
    "central": (25, 30, 29, 34),
    "moving": (35, 40, 14, 19),
    "black": (10, 15, 19, 24),
    "refill": (5, 10, 44, 49),
    "target": (4, 11, 53, 60),
    "hud": (53, 63, 1, 11),
}


def pixels(frame, bounds):
    r0, r1, c0, c1 = bounds
    return "/".join(
        "".join(format(int(value), "x") for value in row)
        for row in np.asarray(frame)[r0:r1, c0:c1]
    )


def inspect(env):
    reach_level_5(env)
    frame = env.frame()
    for name, bounds in OBJECTS.items():
        print(name, pixels(frame, bounds))
    clone = env.clone()
    clone.step(1)
    print("moving_after_one", pixels(clone.frame(), (35, 40, 19, 24)))


if __name__ == "__main__":
    levels, path, error = A.run_program("ls20", inspect)
    print("probe_result", levels, len(path), error)
