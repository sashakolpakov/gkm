"""Sibling-clone isolation + careful re-probe of single clicks."""
import sys, hashlib
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


def h(f):
    return hashlib.md5(np.asarray(f).tobytes()).hexdigest()[:6]


def d(a, b, skip_row1=True):
    a, b = np.asarray(a), np.asarray(b)
    m = (a != b)
    if skip_row1:
        m[1, :] = False
    ys, xs = np.where(m)
    if not len(ys):
        return "SAME"
    return f"n={len(ys)} " + " ".join(f"({y},{x}){int(a[y,x])}->{int(b[y,x])}" for y, x in list(zip(ys, xs))[:20])


def run(env):
    base = np.asarray(env.frame()).copy()
    c1, c2 = env.clone(), env.clone()
    b2 = np.asarray(c2.frame()).copy()
    for _ in range(30):
        c1.step(6, 20, 44)
    print("sibling c2 after c1 took 30 steps:", d(b2, c2.frame(), skip_row1=False))
    print("parent env:", d(base, env.frame(), skip_row1=False))
    print("--- isolated single clicks (fresh clone each, printed in order) ---")
    for pt in [(36, 20), (40, 20), (44, 20), (20, 44), (36, 44), (0, 0), (36, 20)]:
        c = env.clone()
        c.step(6, *pt)
        print(f"  click x={pt[0]:2d} y={pt[1]:2d}:", d(base, c.frame()))


A.run_program("tn36", run)
