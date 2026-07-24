import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ROWS={0:50,1:55,2:60}; COLS={0:25,1:30,2:35}
def center(i,j): return (COLS[j],ROWS[i])
def show(f):
    for r in range(49,62):
        print(f" {r:2d} "+''.join('.' if f[r,c]==0 else format(f[r,c],'x') for c in range(24,37)))
def main(env):
    # click single empty edge (0,1)=(30,50) with lead dummy, trail dummy
    for clicks in [1,2,3]:
        c=env.clone()
        c.step(1) # dummy
        for _ in range(clicks): c.step(6,30,50)
        c.step(1) # reveal
        f=np.asarray(c.frame())
        print(f"empty(30,50) x{clicks} clicks (with dummy): center-val={f[50,30]}")
    print("grid for x1:")
    c=env.clone(); c.step(1); c.step(6,30,50); c.step(1)
    show(np.asarray(c.frame()))
A.run_program('sc25', main)
