import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import block_signatures, connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)


PATH = [
    2, 4,
    (6, 56, 28), 4, (6, 56, 28), 4, (6, 56, 28), 4, (6, 56, 28),
    4, 4, 4, 4, 4, 4,
    2, 2, 2, 2, 4,
]


def summary(frame):
    arr = np.asarray(frame)
    ys, xs = np.where(arr == 14)
    av = None if not len(ys) else (int(ys.min()), int(xs.min()))
    movers = [(b.bbox, b.area) for b in connected_components(frame, colors=(1,), min_area=1)]
    return av, movers


def probe(env):
    solver.solve(env)
    print("S00", summary(env.frame()))
    for i, action in enumerate(PATH, 1):
        before = np.asarray(env.frame()).copy()
        env.step(*action) if isinstance(action, tuple) else env.step(action)
        delta = frame_delta(before, env.frame())
        print(f"S{i:02d}", action, summary(env.frame()), "D", delta["count"], delta["bbox"])
    base = np.asarray(env.frame()).copy()
    base_sigs = block_signatures(base, 2)
    for point in ((52, 19), (46, 28), (56, 28)):
        clone = env.clone()
        clone.step(6, *point)
        changes = {
            k: v for k, v in block_signatures(clone.frame(), 2).items()
            if k[0] < 28 and k[1] < 19 and v != base_sigs[k]
        }
        print("CONTEXT_CLICK", point, "SUMMARY", summary(clone.frame()), "CHANGES", changes)
    mover = env.clone()
    for i in range(1, 9):
        mover.step(6, 46, 28)
        print("SHIFT", i, summary(mover.frame()))
    chars = {1: "B", 2: ".", 3: " ", 4: "#", 5: "5", 11: "G", 12: "C", 14: "A"}
    arr = np.asarray(env.frame())
    for r in range(8, 56, 2):
        row = []
        for c in range(0, 38, 2):
            vals = np.unique(arr[r:r + 2, c:c + 2])
            row.append(chars.get(int(vals[0]), "?") if len(vals) == 1 else "*")
        print(f"T{r // 2:02d}", "".join(row))
    print("LEVEL", int(env.levels_completed))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
