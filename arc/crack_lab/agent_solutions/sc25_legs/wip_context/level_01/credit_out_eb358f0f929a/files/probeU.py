import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
def reservoir(f): return int((f[:,62:64]==14).sum())
def cell(f,r,c): return ''.join('.' if f[r,cc]==0 else format(f[r,cc],'x') for cc in range(c,c+3))
def main(env):
    for N in [1,2,3,4]:
        c=env.clone()
        for i in range(N): c.step(6,30,50)   # click empty middle-top N times
        c.step(1) # settle (up, harmless)
        f=np.asarray(c.frame())
        print(f"empty x{N} clicks: cell={cell(f,50,29)} res={reservoir(f)} lvl={c.levels_completed}")
    print("---filled cell---")
    for N in [1,2,3]:
        c=env.clone()
        for i in range(N): c.step(6,25,50)   # click filled top-left N times
        c.step(1)
        f=np.asarray(c.frame())
        print(f"filled x{N} clicks: cell={cell(f,50,24)} res={reservoir(f)} lvl={c.levels_completed}")
A.run_program('sc25', main)
