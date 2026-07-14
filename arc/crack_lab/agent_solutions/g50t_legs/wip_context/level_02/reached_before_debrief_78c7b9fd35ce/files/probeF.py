import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def prog(env):
    c=env.clone()
    print("start",P.color_counts(c.frame()))
    for a in [4,4,4,4,2,2,2,2,3,3]:
        c.step(a)
        print("act",a,P.color_counts(c.frame()))
    raise SystemExit
A.run_program('g50t', prog)
