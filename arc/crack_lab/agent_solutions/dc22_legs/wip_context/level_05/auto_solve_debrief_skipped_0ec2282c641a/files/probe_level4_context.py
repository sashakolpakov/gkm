import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


STAGE = [
    2, 4,
    (6, 56, 28), 4, (6, 56, 28), 4, (6, 56, 28), 4, (6, 56, 28),
    4, 4, 4, 4, 4, 4,
    2, 2, 2, 2, 4,
]


def step(env, action):
    env.step(*action) if isinstance(action, tuple) else env.step(action)


def compact(frame):
    a = np.asarray(frame)
    ys, xs = np.where(a == 14)
    avatar = None if not len(ys) else (int(ys.min()), int(xs.min()))
    blobs = [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=4)
        if b.color in (1, 6, 7, 8, 9, 11, 12, 13, 14, 15)
    ]
    return avatar, blobs


def scan(env, label):
    base = np.asarray(env.frame()).copy()
    outcomes = {}
    for y in range(8, 56, 2):
        for x in range(40, 64, 2):
            clone = env.clone()
            clone.step(6, x, y)
            play = np.asarray(clone.frame())[:56].tobytes()
            if play != base[:56].tobytes():
                outcomes.setdefault(play, {"points": [], "clone": clone}).get("points").append((x, y))
    print(label, "STATE", compact(base))
    for outcome in outcomes.values():
        delta = frame_delta(base, outcome["clone"].frame())
        print(label, "HIT", outcome["points"], "D", delta["count"], delta["bbox"],
              "SAMPLES", delta["samples"], "AFTER", compact(outcome["clone"].frame()))


def probe(env):
    solver.solve(env)
    scan(env, "START")
    for action in STAGE:
        step(env, action)
    scan(env, "STAGED")


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
