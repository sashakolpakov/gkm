"""Is the world clocked by wall-time or by global step count?

Variant A: few steps, long elapsed time.
Variant B: many steps, short elapsed time.
Each observation = one fresh clone + one neutral step, reporting cyan blobs.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
from perception import connected_components

MODE = sys.argv[1] if len(sys.argv) > 1 else "A"


def cyan(f):
    return [(b.bbox, b.area) for b in connected_components(f, colors=[11], min_area=1)]


def probe(env, t0, nsteps, tag):
    cl = env.clone()
    cl.step(6, 0, 0)
    f = np.asarray(cl.frame())
    print(f"{tag} t={time.time()-t0:5.1f}s steps={nsteps:5d} cyan={cyan(f)}")


def burn(env, n):
    done = 0
    while done < n:
        cl = env.clone()
        for _ in range(min(40, n - done)):
            cl.step(6, 0, 0)
            done += 1
            if cl.terminal():
                break
    return done


def run(env):
    t0 = time.time()
    total = 0
    print("root frame cyan at t0:", cyan(np.asarray(env.frame())))
    if MODE == "A":
        for i in range(16):
            total += 1
            probe(env, t0, total, f"A{i:02d}")
            time.sleep(1.5)
    else:
        for i in range(16):
            total += burn(env, 300)
            total += 1
            probe(env, t0, total, f"B{i:02d}")
    print("root frame cyan at end:", cyan(np.asarray(env.frame())))


A.run_program("tn36", run)
