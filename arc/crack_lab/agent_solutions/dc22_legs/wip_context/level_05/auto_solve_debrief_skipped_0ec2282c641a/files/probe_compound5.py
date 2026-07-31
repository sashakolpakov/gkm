import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

from perception import connected_components, frame_delta


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

C = (6, 50, 28)
D = (6, 56, 28)
DOCK = (6, 52, 35)
X = (6, 44, 29)


def objects(frame):
    return [
        (b.color, b.bbox, b.area)
        for b in connected_components(
            np.asarray(frame)[:56, :38], colors=(8, 12, 15))
    ]


def tile_rows(frame, first=2, last=18):
    symbols = {0: " ", 2: ".", 4: "#", 6: "6", 7: "7", 8: "P",
               9: "=", 11: "G", 12: "C", 14: "A", 15: "!"}
    a = np.asarray(frame)
    rows = []
    for r in range(first, last + 1):
        chars = []
        for c in range(19):
            vals, counts = np.unique(
                a[2 * r:2 * r + 2, 2 * c:2 * c + 2],
                return_counts=True)
            chars.append(symbols.get(
                int(vals[int(np.argmax(counts))]), "?"))
        rows.append(f"{r:02d} {''.join(chars)}")
    return "/".join(rows)


def probe(env):
    solver.solve(env)
    for action in (C, C, C, D, D, D, DOCK):
        env.step(*action)
    before = env.frame()
    print("ROOT", objects(before), tile_rows(before))
    for phase in range(1, 9):
        env.step(*X)
        after = env.frame()
        d = frame_delta(before, after)
        print("X", phase, objects(after), d["count"], d["bbox"],
              int(env.levels_completed), tile_rows(after))
        before = after


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
