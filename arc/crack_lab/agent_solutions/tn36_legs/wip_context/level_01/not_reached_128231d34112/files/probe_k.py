"""Sweep single clicks over board cell centres; report non-budget deltas.

Tries both argument orders (col,row) and (row,col) to fix the convention.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

R0, C0, CELL = 9, 14, 4
NR, NC = 8, 9


def centre(r, c):
    return R0 + 4 * r + 1, C0 + 4 * c + 1


def diff(a, b):
    a = np.asarray(a).copy(); b = np.asarray(b).copy()
    a[1, :] = 0; b[1, :] = 0
    ys, xs = np.where(a != b)
    if len(ys) == 0:
        return None
    return (len(ys), (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
            [(int(y), int(x), int(a[y, x]), int(b[y, x])) for y, x in list(zip(ys, xs))[:8]])


def run(env):
    base = np.asarray(env.frame()).copy()
    t0 = time.time()
    for order in ("xy=col,row", "xy=row,col"):
        print("---", order)
        hits = 0
        for r in range(NR):
            for c in range(NC):
                y, x = centre(r, c)
                cl = env.clone()
                if order.startswith("xy=col"):
                    cl.step(6, x, y)
                else:
                    cl.step(6, y, x)
                d = diff(base, cl.frame())
                if d:
                    hits += 1
                    print(f"  cell({r},{c}) px({y},{x}) n={d[0]} bbox={d[1]} lvl={cl.levels_completed} {d[2][:4]}")
        print("  hits", hits)
    print("elapsed", round(time.time() - t0, 1))


A.run_program("tn36", run)
