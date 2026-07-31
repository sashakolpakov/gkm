import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

CONTROLS = {
    "a": (46, 22),
    "b": (56, 22),
    "c": (50, 28),
    "d": (56, 28),
    "e": (52, 42),
    "f": (52, 46),
}


def avatar(frame):
    ys, xs = np.where(np.asarray(frame) == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def compact(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(frame, min_area=4)
        if b.color not in (0, 3, 4, 5)
    ]


def changed(before, after):
    a, b = np.asarray(before), np.asarray(after)
    mask = (a != b)
    mask[63, :] = False
    out = []
    for color in sorted(set(a[mask].tolist() + b[mask].tolist())):
        ys, xs = np.where(mask & ((a == color) | (b == color)))
        out.append((int(color), (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())), len(ys)))
    return out


def probe(env):
    solver.solve(env)
    print("ACTIONS", env.actions)
    base = env.frame()
    a = np.asarray(base)
    for color in (13,):
        ys, xs = np.where(a == color)
        print("COLOR", color, int(len(ys)),
              list(zip(xs.tolist(), ys.tolist()))[:40])
    print("LEVEL", env.levels_completed, "AV", avatar(base), "OBJECTS", compact(base))
    for name, point in CONTROLS.items():
        clone = env.clone()
        prev = clone.frame()
        for n in range(1, 7):
            clone.step(6, *point)
            now = clone.frame()
            print("CTRL", name, n, "AV", avatar(now), "DELTA", changed(prev, now))
            prev = now
    for act in (1, 2, 3, 4):
        clone = env.clone()
        positions = []
        for _ in range(10):
            clone.step(act)
            positions.append(avatar(clone.frame()))
        print("MOVE", act, positions)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
