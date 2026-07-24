import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ROWS={0:50,1:55,2:60}; COLS={0:25,1:30,2:35}
def center(i,j): return (COLS[j],ROWS[i])
def grid(f):
    return [[int(f[ROWS[i],COLS[j]]) for j in range(3)] for i in range(3)]
def main(env):
    for name,(i,j) in [("center",(1,1)),("corner00",(0,0)),("edgeN",(0,1))]:
        c=env.clone(); c.step(1)
        col,r=center(i,j); c.step(6,col,r)
        c.step(1)
        print(name, grid(np.asarray(c.frame())))
A.run_program('sc25', main)
