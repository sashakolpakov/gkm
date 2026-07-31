import importlib.util
import sys

import numpy as np

sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


spec = importlib.util.spec_from_file_location("solve", "solve.py")
solver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solver)

SYMBOLS = " 123456789ABCDEF"


def probe(env):
    solver.solve(env)
    a = np.asarray(env.frame())
    for r in range(32):
        row = []
        for c in range(19, 32):
            vals, counts = np.unique(
                a[2 * r:2 * r + 2, 2 * c:2 * c + 2],
                return_counts=True)
            row.append(SYMBOLS[int(vals[int(np.argmax(counts))])])
        print(f"{r:02d}", "".join(row))


levels, path, err = A.run_program("dc22", probe)
print("END", levels, len(path), err)
