from l8env import l8
from ferry import ferry_free
from legs import _grid_scan
import numpy as np
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
def gf(f): return int((f[9:15,45:59]==2).sum())+int((f[49:59,49:59]==2).sum())
def stuck(c):
    av,boxes,walls=_grid_scan(c)
    return [b for b in boxes if (b[0]<6 and b[1]<5) or (b[0]>9 and b[1]<6)]
c=l8().clone()
di=0
tdrops=[(4,6),(3,6),(2,6),(4,7)]; bdrops=[(13,7),(12,7),(14,7),(11,7)]
while not c.terminal() and c.levels_completed==7:
    st=stuck(c)
    if not st: break
    st.sort(key=lambda b:(b[1],b[0]))
    box=st[0]
    drop=(tdrops if box[0]<6 else bdrops)[di%4]; di+=1
    ok=ferry_free(c,box,drop)
    print('ferry %s->%s ok=%s mu=%d gf=%d stuckleft=%d lvl=%d'%(box,drop,ok,mu(c),gf(np.asarray(c.frame())),len(stuck(c)),c.levels_completed))
    if not ok:
        # skip by nudging
        c.step(5)
print('ferry done mu',mu(c),'lvl',c.levels_completed)
while not c.terminal() and c.levels_completed==7:
    c.step(5)
    if c.levels_completed>7: print('WIN'); break
print('FINAL lvl',c.levels_completed,'mu',mu(c),'gf',gf(np.asarray(c.frame())))
