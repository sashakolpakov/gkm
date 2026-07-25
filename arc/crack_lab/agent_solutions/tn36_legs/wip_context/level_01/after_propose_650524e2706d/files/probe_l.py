"""Full 64x64 single-click sweep: classify which (a1,a2) pairs do anything."""
import sys, time, collections
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


def sig(a, b):
    a = np.asarray(a).copy(); b = np.asarray(b).copy()
    a[1, :] = 0; b[1, :] = 0
    ys, xs = np.where(a != b)
    if len(ys) == 0:
        return None
    return tuple(sorted((int(y), int(x), int(a[y, x]), int(b[y, x])) for y, x in zip(ys, xs)))


def run(env):
    base = np.asarray(env.frame()).copy()
    t0 = time.time()
    groups = collections.defaultdict(list)
    for a1 in range(64):
        for a2 in range(64):
            cl = env.clone()
            cl.step(6, a1, a2)
            s = sig(base, cl.frame())
            if s is not None:
                groups[s].append((a1, a2))
    print("elapsed", round(time.time() - t0, 1), "distinct effects", len(groups))
    for s, pts in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        a1s = sorted({p[0] for p in pts}); a2s = sorted({p[1] for p in pts})
        print(f"n_px={len(s)} npts={len(pts)} a1={a1s} a2={a2s}")
        print("    ", s[:10])


A.run_program("tn36", run)
