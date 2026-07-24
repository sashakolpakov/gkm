import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

# cell centers (col,row)
ROWS={0:50,1:55,2:60}; COLS={0:25,1:30,2:35}
def center(i,j): return (COLS[j],ROWS[i])
ALL=[(i,j) for i in range(3) for j in range(3)]
EDGES=[(0,1),(1,0),(1,2),(2,1)]
BASE2=[(0,0),(0,2),(1,1),(2,0),(2,2)]

def painted_set(f):
    s=[]
    for i in range(3):
        for j in range(3):
            col,row=center(i,j)
            v=f[row,col]
            if v==14: s.append((i,j))
    return set(s)

def apply(env, targets, lead=1, trail=1):
    c=env.clone()
    for _ in range(lead): c.step(1)  # dummy moves
    for (i,j) in targets:
        col,row=center(i,j); c.step(6,col,row)
    for _ in range(trail): c.step(1)
    return c

def main(env):
    for name,tg in [("edges",EDGES),("all",ALL),("base2",BASE2),("none",[])]:
        c=apply(env,tg)
        f=np.asarray(c.frame())
        print(f"{name}: painted={sorted(painted_set(f))} lvl={c.levels_completed} term={c.terminal()}")
A.run_program('sc25', main)
