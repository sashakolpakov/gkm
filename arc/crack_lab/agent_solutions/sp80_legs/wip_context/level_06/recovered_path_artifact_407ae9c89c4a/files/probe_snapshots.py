"""Compact symbolic snapshots of solved lower-level arrangements."""
import sys

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

import perception as P
from legs import drive_objects


PLANS = {
    1: [],
    2: [
        ((14, 18), [4] * 3),
        ((34, 26), [4] * 2),
        ((30, 38), [4] * 2),
    ],
    3: [
        ((15, 21), [4]),
        ((49, 29), [3] * 10),
        ((19, 33), [4] * 5),
        ((47, 41), [4]),
    ],
    4: [
        ((24, 18), [4] * 3),
        ((45, 18), [3] * 8),
        ((18, 30), [4] * 8),
        ((49, 33), [3] * 10),
        ((43, 42), [3] * 7),
    ],
    5: [
        ((48, 33), [3] * 6),
        ((21, 33), [2] * 3),
        ((34, 21), [4] * 3 + [2] * 7),
        ((36, 45), [3] * 6 + [1] * 2),
    ],
}


def objects(env):
    return [
        (o["color"], o["bbox"], o["area"])
        for o in P.object_candidates(env.frame(), min_area=4)
        if o["color"] in (4, 6, 8, 9, 11, 15)
    ]


def probe(env):
    print("L", 1, "START", objects(env))
    env.step(4)
    env.step(4)
    env.step(4)
    print("L", 1, "PLACED", objects(env))
    env.step(5)
    for level in range(2, 6):
        print("L", level, "START", objects(env))
        drive_objects(env, PLANS[level], commit=[])
        print("L", level, "PLACED", objects(env))
        env.step(5)


if __name__ == "__main__":
    levels, path, err = A.run_program("sp80", probe)
    print("PROBE_RESULT", levels, len(path), err)
