from l8env import l8
from ferry import ferry_free
from legs import _grid_scan
import numpy as np
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
def gf(f): return int((f[9:15,45:59]==2).sum())+int((f[49:59,49:59]==2).sum())
def s2(f): return int((f[9:15,45:59]==2).sum()), int((f[49:59,49:59]==2).sum())
def topstuck(c):
    av,boxes,walls=_grid_scan(c)
    return [b for b in boxes if b[0]<6 and b[1]<5]
c=l8().clone()
drops=[(4,6),(3,6),(4,7),(2,6),(3,7)]; di=0
while c.levels_completed==7 and not c.terminal():
    st=topstuck(c)
    if not st: break
    st.sort(key=lambda b:(b[1],b[0]))
    ok=ferry_free(c,st[0],drops[di%len(drops)]); di+=1
    print('fed %s ok=%s mu=%d s2=%s left=%d'%(st[0],ok,mu(c),s2(np.asarray(c.frame())),len(topstuck(c))))
    if not ok: c.step(5)
print('feeding done mu',mu(c))
mn=999
while c.levels_completed==7 and not c.terminal():
    c.step(5)
    g=gf(np.asarray(c.frame())); mn=min(mn,g)
print('FINAL lvl',c.levels_completed,'mu',mu(c),'s2',s2(np.asarray(c.frame())),'min_gf',mn)
