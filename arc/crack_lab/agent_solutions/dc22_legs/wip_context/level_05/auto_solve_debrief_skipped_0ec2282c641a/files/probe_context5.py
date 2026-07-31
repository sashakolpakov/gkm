import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

CONTROLS = {
    "A": (6, 46, 22),
    "B": (6, 56, 22),
    "C": (6, 50, 28),
    "D": (6, 56, 28),
    "E": (6, 52, 42),
    "F": (6, 52, 46),
}


def bbox(frame, color):
    a = np.asarray(frame)[4:56, :38]
    ys, xs = np.where(a == color)
    return None if not len(ys) else (
        int(ys.min() + 4), int(xs.min()),
        int(ys.max() + 4), int(xs.max()))


def avatar(frame):
    a = np.asarray(frame)
    ys, xs = np.where(a == 14)
    return None if not len(ys) else (int(ys.min()), int(xs.min()))


def run(env, label, prefix):
    root = env.clone()
    for action in prefix:
        root.step(*action) if isinstance(action, tuple) else root.step(action)
    before = root.frame()
    target_count = int((np.asarray(before)[4:56, :38] == 15).sum())
    print("ROOT", label, avatar(before), bbox(before, 8), target_count)
    a = np.asarray(before)
    for r in range(10, 22):
        print("MASK", label, r, "".join(
            "P" if a[r, c] == 8 else
            "T" if a[r, c] == 15 else
            "#" if a[r, c] == 4 else "."
            for c in range(4, 20)))
    for name, action in list(CONTROLS.items()) + [
            ("UP", 1), ("DOWN", 2), ("LEFT", 3), ("RIGHT", 4)]:
        clone = root.clone()
        clone.step(*action) if isinstance(action, tuple) else clone.step(action)
        after = clone.frame()
        d = frame_delta(before, after)
        print("ACT", label, name, d["count"], d["bbox"],
              avatar(after), bbox(after, 8),
              int((np.asarray(after)[4:56, :38] == 15).sum()),
              int(clone.levels_completed))


def probe(env):
    solver.solve(env)
    c = CONTROLS["C"]
    d = CONTROLS["D"]
    run(env, "TARGET", [c, c, c, d])
    run(env, "PAST", [c, c, c, d, d])
    a_ctl = CONTROLS["A"]
    b_ctl = CONTROLS["B"]
    e = CONTROLS["E"]
    run(env, "RIGHT_E", [d, d, d, e])
    run(env, "RIGHT_A", [a_ctl] * 5 + [d, d, d])
    run(env, "RIGHT_B", [b_ctl] * 4 + [d, d, d])
    platform_glyph = (6, 52, 35)
    run(env, "DOCK_ACTIVATE",
        [c, c, c, d, d, d, platform_glyph])


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
