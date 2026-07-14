import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
import numpy as np

def main(env):
    for a in [1,2,3,4,6]:
        clone = env.clone()
        before = np.asarray(clone.frame()).copy()
        if a==6:
            clone.step(6,0,0)
        else:
            clone.step(a)
        d = P.frame_delta(before, clone.frame())
        print("action",a,"count",d['count'],"bbox",d['bbox'])
        print("  samples", d['samples'][:6])

A.run_program('sc25', main)
