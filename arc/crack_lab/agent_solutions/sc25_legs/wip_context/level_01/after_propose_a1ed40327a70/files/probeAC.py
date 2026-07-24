import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ROWS={0:50,1:55,2:60}; COLS={0:25,1:30,2:35}
def center(i,j): return (COLS[j],ROWS[i])
def grid(f):
    g=[]
    for i in range(3):
        row=[]
        for j in range(3):
            col,r=center(i,j); row.append(int(f[r,col]))
        g.append(row)
    return g
EDGES=[(0,1),(1,0),(1,2),(2,1)]
def main(env):
    for n in range(0,5):
        c=env.clone(); c.step(1)
        for k in range(n):
            i,j=EDGES[k]; col,r=center(i,j); c.step(6,col,r)
        c.step(1)
        print(f"after clicking first {n} edges:", grid(np.asarray(c.frame())))
A.run_program('sc25', main)
