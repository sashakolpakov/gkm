import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
ROWS={0:50,1:55,2:60}; COLS={0:25,1:30,2:35}
def center(i,j): return (COLS[j],ROWS[i])
EDGES=[(0,1),(1,0),(1,2),(2,1)]
def main(env):
    base=np.asarray(env.frame()).copy()
    c=env.clone(); c.step(1)
    for (i,j) in EDGES:
        col,r=center(i,j); c.step(6,col,r)
    c.step(1)
    f=np.asarray(c.frame())
    print("lvl",c.levels_completed,"term",c.terminal())
    # diff vs base excluding checkerboard region and timer
    d=f.copy().astype(int)-base.astype(int)
    ys,xs=np.where(f!=base)
    # group changed regions outside checkerboard(rows47-63 cols22-38) and timer(cols60+)
    outside=[(int(y),int(x),int(base[y,x]),int(f[y,x])) for y,x in zip(ys,xs)
             if not(47<=y<=63 and 22<=x<=38) and x<60]
    print("changes outside checkerboard/timer:",len(outside))
    for o in outside[:20]: print("  ",o)
    print("reservoir",int((f[:,62:64]==14).sum()))
A.run_program('sc25', main)
