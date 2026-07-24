import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ROWS={0:50,1:55,2:60}; COLS={0:25,1:30,2:35}
def center(i,j): return (COLS[j],ROWS[i])
def show(f):
    for r in range(49,62):
        print(f" {r:2d} "+''.join('.' if f[r,c]==0 else format(f[r,c],'x') for c in range(24,37)))
EDGES=[(0,1),(1,0),(1,2),(2,1)]
def main(env):
    c=env.clone()
    c.step(1) # dummy
    for (i,j) in EDGES:
        col,row=center(i,j); c.step(6,col,row)
    c.step(1) # reveal
    print("edges clicked once each:")
    show(np.asarray(c.frame()))
    # try with a settle between each click
    print("edges with dummy between clicks:")
    c=env.clone(); c.step(1)
    for (i,j) in EDGES:
        col,row=center(i,j); c.step(6,col,row); c.step(1)
    f=np.asarray(c.frame()); show(f)
A.run_program('sc25', main)
