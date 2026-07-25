"""Instrumented replay of the probe_c sweep to explain the phantom board delta."""
import sys
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


def budget(f):
    return int((np.asarray(f)[1, 1:62] == 9).sum())


def glyphs(f):
    f = np.asarray(f)
    return tuple(int(f[44, c]) for c in (21, 26, 31, 36, 41))


def sig(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = (a != b); m[1, :] = False
    ys, xs = np.where(m)
    return [(int(y), int(x), int(a[y, x]), int(b[y, x])) for y, x in zip(ys, xs)]


def run(env):
    base = np.asarray(env.frame()).copy()
    coords = list(range(0, 64, 4))
    i = -1
    for a_ in coords:
        for b_ in coords:
            i += 1
            c = env.clone()
            c.step(6, a_, b_)
            cf = np.asarray(c.frame()).copy()
            s = sig(base, cf)
            interesting = 138 <= i <= 162
            if s and (interesting or i < 145):
                print(f"i={i:3d} x={a_:2d} y={b_:2d} n={len(s)} rootB={budget(env.frame())} "
                      f"rootG={glyphs(env.frame())} cB={budget(cf)} cG={glyphs(cf)} {s[:8]}")
            elif interesting:
                print(f"i={i:3d} x={a_:2d} y={b_:2d} SAME rootB={budget(env.frame())} cG={glyphs(cf)}")
    print("root at end: budget", budget(env.frame()), "glyphs", glyphs(env.frame()),
          "sig vs base", len(sig(base, env.frame())))


A.run_program("tn36", run)
