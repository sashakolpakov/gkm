import sys, itertools
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ROWS={0:50,1:55,2:60}; COLS={0:25,1:30,2:35}
def center(i,j): return (COLS[j],ROWS[i])
CELLS=[(i,j) for i in range(3) for j in range(3)]
def main(env):
    wins=[]
    for mask in range(512):
        S=[CELLS[k] for k in range(9) if mask&(1<<k)]
        c=env.clone(); c.step(1)
        for (i,j) in S:
            col,r=center(i,j); c.step(6,col,r)
        c.step(1)
        if c.levels_completed>0:
            wins.append(S)
    print("winning subsets:",len(wins))
    for w in wins[:10]: print(w)
A.run_program('sc25', main)
