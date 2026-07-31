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
SYMBOLS = {
    0: " ",
    1: "b",
    2: ".",
    3: " ",
    4: "#",
    5: "x",
    6: "6",
    7: "7",
    8: "P",
    9: "=",
    10: "-",
    11: "G",
    12: "C",
    13: "D",
    14: "A",
    15: "!",
}


def tiles(frame):
    grid = []
    a = np.asarray(frame)
    for r in range(0, 62, 2):
        row = []
        for c in range(0, 38, 2):
            vals, counts = np.unique(a[r:r + 2, c:c + 2], return_counts=True)
            color = int(vals[int(np.argmax(counts))])
            row.append(color)
        grid.append(row)
    return grid


def print_map(frame):
    print("   " + "".join(str(i // 10) if i >= 10 else " " for i in range(19)))
    print("   " + "".join(str(i % 10) for i in range(19)))
    for i, row in enumerate(tiles(frame), 2):
        print(f"{i:02d} " + "".join(SYMBOLS[v] for v in row))


def probe(env):
    solver.solve(env)
    base = tiles(env.frame())
    print_map(env.frame())
    for name, action in CONTROLS.items():
        clone = env.clone()
        for phase in range(1, 7):
            clone.step(*action)
            current = tiles(clone.frame())
            changed = [
                (r + 2, c, SYMBOLS[base[r][c]], SYMBOLS[current[r][c]])
                for r in range(len(base))
                for c in range(len(base[r]))
                if current[r][c] != base[r][c]
            ]
            print(name, phase, changed)


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
