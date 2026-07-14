import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def prog(env):
    print("actions:", env.actions)
    f = np.asarray(env.frame())
    print("shape", f.shape, "levels", env.levels_completed, "terminal", env.terminal())
    print("colors:", P.color_counts(f))
    # print full grid compactly
    for row in f:
        print(''.join(str(int(v)) for v in row))
    raise SystemExit  # stop early

A.run_program('g50t', prog)
