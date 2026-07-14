import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ROWS={0:50,1:55,2:60}; COLS={0:25,1:30,2:35}
def center(i,j): return (COLS[j],ROWS[i])
EDGES=[(0,1),(1,0),(1,2),(2,1)]
def main(env):
    c=env.clone(); c.step(1)
    for (i,j) in EDGES:
        col,r=center(i,j); c.step(6,col,r)
    for t in range(15):
        c.step(1)
        if c.levels_completed>0:
            print("WIN after",t,"trailing"); return
        if c.terminal():
            print("terminal at trailing",t,"lvl",c.levels_completed); return
    print("no win, lvl",c.levels_completed)
    # try also with base2 set + trailing
    BASE2=[(0,0),(0,2),(1,1),(2,0),(2,2)]
    c=env.clone(); c.step(1)
    for (i,j) in BASE2:
        col,r=center(i,j); c.step(6,col,r)
    for t in range(15):
        c.step(1)
        if c.levels_completed>0: print("BASE2 WIN trailing",t); return
    print("base2 no win lvl",c.levels_completed)
A.run_program('sc25', main)
