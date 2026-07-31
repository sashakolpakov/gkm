import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


def compact_map(frame):
    a = np.asarray(frame)
    rows = []
    for r in range(0, 64, 4):
        row = []
        for c in range(0, 64, 4):
            vals, counts = np.unique(a[r:r + 4, c:c + 4], return_counts=True)
            row.append(f"{int(vals[counts.argmax()]):x}")
        rows.append("".join(row))
    return rows


def probe(env):
    solver.solve(env)
    print("LEVEL", env.levels_completed, "ACTIONS", env.actions)
    print("COUNTS", {
        int(v): int(n)
        for v, n in zip(*np.unique(np.asarray(env.frame()), return_counts=True))
    })
    blobs = connected_components(env.frame(), min_area=4)
    print("BLOBS", [
        (b.color, b.bbox, b.area)
        for b in blobs if b.color != 3 or b.area < 1000
    ])
    print("MAP")
    print("\n".join(compact_map(env.frame())))
    base = np.asarray(env.frame()).copy()
    for action in (1, 2, 3, 4):
        clone = env.clone()
        clone.step(action)
        d = frame_delta(base, clone.frame())
        print("ACT", action, "REWARD", clone.levels_completed,
              "DELTA", (d["count"], d["bbox"], d["samples"][:8]))
    controls = ((51, 18), (51, 27), (51, 36))
    for point in controls:
        clone = env.clone()
        clone.step(6, *point)
        d = frame_delta(base, clone.frame())
        print("CLICK", point, "REWARD", clone.levels_completed,
              "DELTA", (d["count"], d["bbox"], d["samples"][:24]))
        print("CLICKMAP", point)
        print("\n".join(compact_map(clone.frame())[3:13]))
    for mask in range(1, 8):
        clone = env.clone()
        seq = []
        for i, point in enumerate(controls):
            if mask & (1 << i):
                clone.step(6, *point)
                seq.append(point)
        print("COMBO", seq, "REWARD", clone.levels_completed)
        print("\n".join(compact_map(clone.frame())[3:13]))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
