"""Re-run the 64x64 sweep with the loop order TRANSPOSED.

If the cyan #1 vanish is coordinate-driven it tracks a1; if it is cumulative it
appears at a fixed sweep index.  Prints the sweep index of every change in the
cyan signature of a freshly cloned single-click probe.
"""
import sys, time
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
from perception import connected_components

ORDER = sys.argv[1] if len(sys.argv) > 1 else "a2_outer"


def cy(f):
    return tuple(b.bbox for b in connected_components(f, colors=[11], min_area=1))


def run(env):
    t0 = time.time()
    prev = cy(np.asarray(env.frame()))
    print("base", prev)
    i = 0
    pairs = ([(a1, a2) for a2 in range(64) for a1 in range(64)] if ORDER == "a2_outer"
             else [(a1, a2) for a1 in range(64) for a2 in range(64)])
    for a1, a2 in pairs:
        i += 1
        cl = env.clone()
        cl.step(6, a1, a2)
        c = cy(np.asarray(cl.frame()))
        if c != prev:
            print(f"i={i:5d} a1={a1:2d} a2={a2:2d} t={time.time()-t0:5.1f}s cyan={c}")
            prev = c
    print("end root", cy(np.asarray(env.frame())), "t", round(time.time() - t0, 1))


A.run_program("tn36", run)
