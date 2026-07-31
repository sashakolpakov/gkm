import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


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
C = CONTROLS["C"]
D = CONTROLS["D"]
CONTEXTS = {
    "BASE": (),
    "TARGET": (C, C, C, D),
    "DOCK": (C, C, C, D, D, D),
}


def first_point(frame, color):
    a = np.asarray(frame)[:56, :38]
    ys, xs = np.where(a == color)
    return int(xs[0]), int(ys[0])


def probe(env):
    solver.solve(env)
    for label, path in CONTEXTS.items():
        root = env.clone()
        for action in path:
            root.step(*action)
        frame = root.frame()
        points = {
            "avatar": first_point(frame, 14),
            "platform": first_point(frame, 8),
            "portal": first_point(frame, 6),
            "bridge": first_point(frame, 9),
            "goal": first_point(frame, 11),
            "switch": first_point(frame, 15),
            "platform_glyph": (52, 35),
            "wall": (1, 5),
            "floor": first_point(frame, 2),
        }
        direct = {}
        for name, control in CONTROLS.items():
            child = root.clone()
            child.step(*control)
            direct[name] = np.asarray(child.frame())[:56].copy()
        hits = []
        for point_name, point in points.items():
            for control_name, control in CONTROLS.items():
                child = root.clone()
                child.step(6, *point)
                child.step(*control)
                after = np.asarray(child.frame())[:56]
                if not np.array_equal(after, direct[control_name]):
                    mask = after != direct[control_name]
                    ys, xs = np.where(mask)
                    hits.append((
                        point_name, point, control_name, int(mask.sum()),
                        (int(ys.min()), int(xs.min()),
                         int(ys.max()), int(xs.max()))))
        print("CONTEXT", label, "HITS", hits)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
