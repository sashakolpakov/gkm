import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def avbb(f):
    pts=[(r,cc) for r in range(15,25) for cc in range(10,60) if f[r,cc] in(9,10)]
    if not pts: return None
    rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
    return (min(rs),min(cs),max(rs),max(cs))

def main(env):
    # go leftmost (need extra step for lag): press left ~7
    c=env.clone()
    for i in range(7): c.step(3)
    print("at leftmost avbb", avbb(np.asarray(c.frame())))
    # now try up 8
    c2=c.clone()
    for i in range(8): c2.step(1)
    print("after up8 avbb", avbb(np.asarray(c2.frame())),"lvl",c2.levels_completed)
    c3=c.clone()
    for i in range(8): c3.step(2)
    print("after down8 avbb", avbb(np.asarray(c3.frame())),"lvl",c3.levels_completed)
A.run_program('sc25', main)
