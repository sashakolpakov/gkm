"""Symbolic transition model probe for lp85 level 7."""
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components
from solve import solve

CONTROLS = ((22, 34), (32, 42))
H = tuple((23, c) for c in range(20, 44, 3))
V = tuple((r, 23) for r in range(20, 44, 3))
Q = ((29, 35), (29, 38), (32, 35), (32, 38))
ANCHORS = ((20, 26), (20, 29))


def sample(env):
    a = np.asarray(env.frame())
    read = lambda ps: "".join(f"{int(a[p]):X}" for p in ps)
    return read(H), read(V), read(Q), read(ANCHORS)


def tokens(env):
    return tuple(
        (b.top_left, b.color)
        for b in connected_components(env.frame(), min_area=4)
        if b.area == 4 and b.size == (2, 2) and b.bbox[0] > 10
    )


def run(env):
    solve(env)
    print("base", sample(env), tokens(env))
    for control in CONTROLS:
        clone = env.clone()
        print("repeat", control)
        for n in range(1, 17):
            clone.step(6, *control)
            print(n, sample(clone), tokens(clone),
                  "level", clone.levels_completed)
            if clone.levels_completed > env.levels_completed:
                break
    print("alternating")
    clone = env.clone()
    for n in range(1, 17):
        control = CONTROLS[(n - 1) % 2]
        clone.step(6, *control)
        print(n, control, sample(clone), "level", clone.levels_completed)
        if clone.levels_completed > env.levels_completed:
            break


if __name__ == "__main__":
    A.run_program("lp85", run)
