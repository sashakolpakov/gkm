import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
def avcol(f):
    cols=[c for c in range(17,60) if f[20,c] in(9,10)]
    return min(cols) if cols else None
def main(env):
    for K in range(6):
        c=env.clone()
        for i in range(K): c.step(3) # left
        print(f"after {K} lefts: avatar-leftcol={avcol(np.asarray(c.frame()))}")
A.run_program('sc25', main)
