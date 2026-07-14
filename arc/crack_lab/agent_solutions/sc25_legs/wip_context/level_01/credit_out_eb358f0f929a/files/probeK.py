import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def avcol(f):
    # find leftmost col in rows19-22 that is 9 or 10
    for c in range(10,60):
        if f[19,c] in (9,10) and f[19,c-1]==2 or (f[19,c] in(9,10) and c<44 and c>17):
            pass
    cols=[c for c in range(17,60) if f[20,c] in (9,10)]
    return (min(cols),max(cols)) if cols else None

def main(env):
    c=env.clone()
    for i in range(10):
        f=np.asarray(c.frame())
        print(f"before step {i}: avcol={avcol(f)}")
        c.step(3) # left
A.run_program('sc25', main)
