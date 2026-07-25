"""Map, over the whole (a1,a2) click plane, what a single click does.

Prints a 64x64 ASCII map keyed by effect class so the geometry of the action
surface is visible at a glance.
"""
import sys, time, collections
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


def eff(base, f):
    a = base.copy(); b = np.asarray(f).copy()
    a[1, :] = 0; b[1, :] = 0
    ys, xs = np.where(a != b)
    if len(ys) == 0:
        return ()
    return tuple(sorted((int(y), int(x), int(a[y, x]), int(b[y, x])) for y, x in zip(ys, xs)))


def run(env):
    t0 = time.time()
    base = np.asarray(env.frame()).copy()
    grid = {}
    classes = collections.OrderedDict()
    for a2 in range(64):
        for a1 in range(64):
            cl = env.clone()
            cl.step(6, a1, a2)
            e = eff(base, cl.frame())
            if e not in classes:
                classes[e] = chr(ord('a') + len(classes)) if e else '.'
            grid[(a1, a2)] = classes[e]
    print("elapsed", round(time.time() - t0, 1), "classes", len(classes))
    print("     " + "".join(str(a1 % 10) for a1 in range(64)))
    for a2 in range(64):
        print(f" {a2:2d}  " + "".join(grid[(a1, a2)] for a1 in range(64)))
    for e, ch in classes.items():
        if not e:
            continue
        colors = collections.Counter((v[2], v[3]) for v in e)
        r0 = min(v[0] for v in e); r1 = max(v[0] for v in e)
        c0 = min(v[1] for v in e); c1 = max(v[1] for v in e)
        print(f"{ch}: n={len(e)} bbox=({r0},{c0},{r1},{c1}) trans={dict(colors)}")


A.run_program("tn36", run)
