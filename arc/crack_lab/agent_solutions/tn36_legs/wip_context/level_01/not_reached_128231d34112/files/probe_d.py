"""Isolate what advances tn36 state: wall clock, clone count, or steps."""
import sys, time, hashlib
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A
from perception import connected_components

MODE = sys.argv[1]


def cy(f):
    return [b.bbox for b in connected_components(f, colors=[11], min_area=2)]


def h(f):
    return hashlib.md5(np.asarray(f).tobytes()).hexdigest()[:6]


def report(tag, t0, env):
    f = np.asarray(env.frame())
    print(f"{tag:>26s} t={time.time()-t0:6.2f} hash={h(f)} cyan={cy(f)} lvl={env.levels_completed}")


def run(env):
    t0 = time.time()
    report("start", t0, env)
    if MODE == "clock":
        for k in range(6):
            spin = time.time()
            while time.time() - spin < 1.0:
                pass
            report(f"after {k+1}s idle", t0, env)
    elif MODE == "clones":
        for k in range(6):
            for _ in range(50):
                env.clone()
            report(f"after {(k+1)*50} clones", t0, env)
    elif MODE == "clonesteps":
        for k in range(6):
            for _ in range(50):
                c = env.clone()
                c.step(6, 0, 0)
            report(f"after {(k+1)*50} clonesteps", t0, env)
    elif MODE == "framecalls":
        for k in range(6):
            for _ in range(200):
                env.frame()
            report(f"after {(k+1)*200} frames", t0, env)
    elif MODE == "realsteps":
        for k in range(6):
            env.step(6, 0, 0)
            report(f"after {k+1} real steps", t0, env)


A.run_program("tn36", run)
