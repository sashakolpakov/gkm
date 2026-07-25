"""Does repeated cloning drift the CLONES (constant click, 300 reps)?"""
import sys, gc
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

HOLD = "hold" in sys.argv


def sig(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = (a != b); m[1, :] = False
    ys, xs = np.where(m)
    return tuple((int(y), int(x), int(a[y, x]), int(b[y, x])) for y, x in zip(ys, xs))


def run(env):
    base = np.asarray(env.frame()).copy()
    held = []
    prev = None
    for i in range(300):
        c = env.clone()
        c.step(6, 36, 20)
        s = sig(base, c.frame())
        if HOLD:
            held.append(c)
        if s != prev:
            print(f"  i={i:3d} nchanged={len(s)} first={s[:6]}")
            prev = s
    print("  parent unchanged:", len(sig(base, env.frame())) == 0, "held", len(held))


A.run_program("tn36", run)
