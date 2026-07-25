"""Locate the cumulative threshold at which fresh clones stop showing cyan blob #1."""
import sys, time
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
from perception import connected_components

MODE = sys.argv[1] if len(sys.argv) > 1 else "step"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096


def cyan(f):
    return [(b.bbox, b.area) for b in connected_components(f, colors=[11], min_area=1)]


def run(env):
    t0 = time.time()
    prev = None
    for i in range(1, N + 1):
        cl = env.clone()
        if MODE == "step":
            cl.step(6, 0, 0)
        f = np.asarray(cl.frame())
        c = cyan(f)
        if c != prev:
            print(f"i={i:5d} t={time.time()-t0:5.1f}s cyan={c}")
            prev = c
    print(f"done i={N} t={time.time()-t0:5.1f}s root_cyan={cyan(np.asarray(env.frame()))}")


A.run_program("tn36", run)
