import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

C = (6, 50, 28)
D = (6, 56, 28)
SYMBOLS = {0: " ", 2: ".", 4: "#", 6: "6", 7: "7", 8: "P",
           9: "=", 11: "G", 12: "C", 14: "A", 15: "!"}


def tile_rows(frame, first=3, last=16):
    a = np.asarray(frame)
    rows = []
    for r in range(first, last + 1):
        chars = []
        for c in range(19):
            vals, counts = np.unique(
                a[2 * r:2 * r + 2, 2 * c:2 * c + 2],
                return_counts=True)
            chars.append(SYMBOLS.get(
                int(vals[int(np.argmax(counts))]), "?"))
        rows.append(f"{r:02d} {''.join(chars)}")
    return "/".join(rows)


def probe(env):
    solver.solve(env)
    paths = {
        "C3D3": [C] * 3 + [D] * 3,
        "D3C3": [D] * 3 + [C] * 3,
        "C2D2": [C] * 2 + [D] * 2,
        "D2C2": [D] * 2 + [C] * 2,
        "C3D2": [C] * 3 + [D] * 2,
        "D2C3": [D] * 2 + [C] * 3,
    }
    for label, path in paths.items():
        clone = env.clone()
        for action in path:
            clone.step(*action)
        a = np.asarray(clone.frame())
        print(label, "COUNTS",
              tuple((color, int((a == color).sum()))
                    for color in (2, 6, 7, 8, 11, 12, 15)),
              "MAP", tile_rows(a))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
