from l8env import l8
from ferry import ferry_free
from legs import _grid_scan
import numpy as np
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
def boxcells(f):
    out=set()
    for R in range(16):
        for C in range(16):
            u=set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
            if 9 in u and ({4,5,3,0}&u) and 2 not in u: out.add((R,C))
    return out
# placed areas: goals + regions (position-based). floor = box outside these.
def is_placed(R,C):
    if 2<=R<=3 and 11<=C<=14: return True   # top goal
    if 12<=R<=14 and 12<=C<=14: return True  # bottom goal
    if (R,C) in {(2,3),(3,3),(4,1),(4,2),(4,3)}: return True  # top region
    if 12<=R<=14 and 3<=C<=5: return True    # bottom region
    return False
def floor(f): return [b for b in boxcells(f) if not is_placed(*b)]
c=l8().clone()
# ferry cluster into top region cells
tgts=[(4,1),(4,2),(4,3),(2,3),(3,3)]; i=0
for _ in range(6):
    av,boxes,w=_grid_scan(c)
    st=[b for b in boxes if not is_placed(*b) and b[0]<6]
    if not st: break
    st.sort()
    ferry_free(c,st[0],tgts[i%len(tgts)]); i+=1
    if c.terminal():break
print('cluster placed mu',mu(c),'floor',sorted(floor(np.asarray(c.frame()))))
# idle, watch floor count and lvl
minf=99
for t in range(200):
    if c.terminal() or c.levels_completed>7: break
    c.step(5)
    fl=len(floor(np.asarray(c.frame()))); minf=min(minf,fl)
    if fl<=1 or c.levels_completed>7:
        print('t%d floor=%d lvl=%d mu=%d'%(t,fl,c.levels_completed,mu(c)))
print('FINAL lvl',c.levels_completed,'minfloor',minf,'mu',mu(c))
