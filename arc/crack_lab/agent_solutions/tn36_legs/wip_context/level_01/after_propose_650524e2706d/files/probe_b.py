import sys, hashlib, inspect
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


def h(f):
    return hashlib.md5(np.asarray(f).tobytes()).hexdigest()[:8]


def diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ys, xs = np.where(a != b)
    if len(ys) == 0:
        return "SAME"
    return f"n={len(ys)} bbox=({ys.min()},{xs.min()},{ys.max()},{xs.max()}) " + \
        " ".join(f"({y},{x}){int(a[y,x])}->{int(b[y,x])}" for y, x in list(zip(ys, xs))[:12])


def run(env):
    print("step sig:", inspect.signature(env.step))
    base = np.asarray(env.frame()).copy()
    print("base hash", h(base))
    # 1. stability of frame() without acting
    for i in range(3):
        print(f"  reread {i}: ", diff(base, env.frame()))
    # 2. clone with no action
    c = env.clone()
    print("  fresh clone no-op:", diff(base, c.frame()))
    # 3. does a nested clone chain drift?
    c2 = env.clone().clone().clone()
    print("  clone^3 no-op:", diff(base, c2.frame()))
    # 4. bad coords / arity
    for args in [(6,), (6, 0, 0), (6, 63, 63), (6, 100, 100), (6, -1, -1)]:
        c = env.clone()
        try:
            c.step(*args)
            print("  step", args, "->", diff(base, c.frame()), "lvl", c.levels_completed, "term", c.terminal())
        except Exception as e:
            print("  step", args, "-> EXC", type(e).__name__, str(e)[:120])


A.run_program("tn36", run)
