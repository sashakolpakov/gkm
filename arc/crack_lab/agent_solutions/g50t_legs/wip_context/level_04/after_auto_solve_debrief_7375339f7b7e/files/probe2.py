import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def prog(env):
    d = P.action_deltas(env)
    for a,info in d.items():
        print(P.ACTION_NAME[a], info['count'], info['bbox'], info['samples'][:6])
    raise SystemExit
A.run_program('g50t', prog)
