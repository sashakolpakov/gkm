"""Inspect env.path aliasing across clones -- likely source of the sweep drift."""
import sys
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


def cls(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = (a != b); m[1, :] = False
    ys, xs = np.where(m)
    return "-" if not len(ys) else f"{len(ys)}@({ys.min()},{xs.min()})-({ys.max()},{xs.max()})"


def run(env):
    base = np.asarray(env.frame()).copy()
    print("root path:", type(env.path), env.path)
    c = env.clone()
    print("clone path is root path?", c.path is env.path)
    c.step(6, 20, 44)
    print("after clone step -> clone path:", c.path, " root path:", env.path)
    coords = list(range(0, 64, 4))
    pts = [(a, b) for a in coords for b in coords]
    for i, (x, y) in enumerate(pts):
        cc = env.clone()
        cc.step(6, x, y)
        k = cls(base, cc.frame())
        if k != "-" or i in (0, 50, 82, 83, 100, 142, 143):
            print(f"i={i:3d} ({x},{y}) {k} rootpathlen={len(env.path)} clonepathlen={len(cc.path)}")
        if i > 150:
            break


A.run_program("tn36", run)
