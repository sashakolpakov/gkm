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
def is_placed(R,C):
    if 2<=R<=3 and 11<=C<=14: return True
    if 12<=R<=14 and 12<=C<=14: return True
    if (R,C) in {(2,3),(3,3),(4,1),(4,2),(4,3)}: return True
    if 12<=R<=14 and 3<=C<=5: return True
    return False
def floor(f): return [b for b in boxcells(f) if not is_placed(*b)]
c=l8().clone()
td=[(4,6),(3,6),(4,7),(2,6),(3,7)]; i=0
for _ in range(6):
    st=[b for b in _grid_scan(c)[1] if b[0]<6 and b[1]<5]
    if not st: break
    st.sort(); ferry_free(c,st[0],td[i%5]); i+=1
    if c.terminal(): break
print('fed mu',mu(c),'floor',len(floor(np.asarray(c.frame()))))
minf=99; winat=None
for t in range(300):
    if c.terminal() or c.levels_completed>7: break
    c.step(5)
    fl=floor(np.asarray(c.frame())); minf=min(minf,len(fl))
    if len(fl)<=2:
        print('t%d floor=%s lvl=%d mu=%d'%(t,sorted(fl),c.levels_completed,mu(c)))
    if c.levels_completed>7: winat=t; break
print('FINAL lvl',c.levels_completed,'minfloor',minf,'mu',mu(c),'winat',winat)
