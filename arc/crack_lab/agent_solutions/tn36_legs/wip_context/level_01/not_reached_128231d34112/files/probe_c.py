import sys
import numpy as np
sys.path.insert(0, "/Users/sasha/gkm/arc/crack_lab")
import gkm_arena as A

BUDGET = [(1, c) for c in range(1, 62)]


def sig(base, after):
    """Delta ignoring the top action-budget bar (row 1)."""
    a, b = np.asarray(base), np.asarray(after)
    d = (a != b)
    d[1, :] = False
    ys, xs = np.where(d)
    if len(ys) == 0:
        return None
    return (len(ys), int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))


def run(env):
    base = np.asarray(env.frame()).copy()
    hits = {}
    stride = 4
    coords = list(range(0, 64, stride))
    grid = {}
    for a_ in coords:
        for b_ in coords:
            c = env.clone()
            c.step(6, a_, b_)
            s = sig(base, c.frame())
            grid[(a_, b_)] = s
            if s:
                hits[(a_, b_)] = (s, int(c.levels_completed))
    print("non-budget-changing clicks:", len(hits), "of", len(grid))
    for k in sorted(hits):
        print("  step(6,%d,%d)" % k, hits[k])
    # also small integer coords (cell-index hypothesis)
    print("--- small coords 0..9 ---")
    small = {}
    for a_ in range(10):
        for b_ in range(10):
            c = env.clone()
            c.step(6, a_, b_)
            s = sig(base, c.frame())
            if s:
                small[(a_, b_)] = (s, int(c.levels_completed))
    print("hits:", len(small))
    for k in sorted(small):
        print("  step(6,%d,%d)" % k, small[k])


A.run_program("tn36", run)
