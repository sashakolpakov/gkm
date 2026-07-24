import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
def reservoir(f): return int((f[:,62:64]==14).sum())
def cell(f,r,c): return ''.join('.' if f[r,cc]==0 else format(f[r,cc],'x') for cc in range(c,c+3))
def main(env):
    print("EMPTY cell (30,50):")
    for K in range(1,6):
        c=env.clone()
        for i in range(K): c.step(6,30,50)
        f=np.asarray(c.frame())
        print(f" K={K} (eff={K-1}) cell={cell(f,50,29)} res={reservoir(f)}")
    print("FILLED-2 cell (25,50):")
    for K in range(1,6):
        c=env.clone()
        for i in range(K): c.step(6,25,50)
        f=np.asarray(c.frame())
        print(f" K={K} (eff={K-1}) cell={cell(f,50,24)} res={reservoir(f)}")
A.run_program('sc25', main)
