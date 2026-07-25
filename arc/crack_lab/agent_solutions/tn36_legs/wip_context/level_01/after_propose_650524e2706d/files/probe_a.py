import sys
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A


def show(env):
    print("actions:", env.actions)
    print("public attrs:", [a for a in dir(env) if not a.startswith('_')])
    f = np.asarray(env.frame())
    print("shape", f.shape, "dtype", f.dtype)
    vals, cnts = np.unique(f, return_counts=True)
    print("colors", dict(zip(vals.tolist(), cnts.tolist())))
    print("levels_completed", env.levels_completed, "terminal", env.terminal())
    # compact ASCII dump with a legend
    legend = {0: '.', 1: '1', 4: '4', 5: '5', 9: '9', 11: 'C'}
    print("   " + "".join(str(c % 10) for c in range(f.shape[1])))
    for r in range(f.shape[0]):
        print(f"{r:2d} " + "".join(legend.get(int(v), '?') for v in f[r]))


A.run_program("tn36", show)
