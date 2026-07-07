from l8env import l8
from ferry import ferry_free
from legs import _grid_scan
import numpy as np
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
def s2(f): return int((f[9:15,45:59]==2).sum()), int((f[49:59,49:59]==2).sum())
topg=set((r,co) for r in (2,3) for co in (11,12,13,14))
botg=set((r,co) for r in (12,13,14) for co in (12,13,14))
def boxes(c):
    av,b,w=_grid_scan(c); return b
c=l8().clone()
# Phase 1: feed top cluster
tdrops=[(4,6),(3,6),(4,7),(2,6),(3,7)]; di=0
for _ in range(6):
    st=[b for b in boxes(c) if b[0]<6 and b[1]<5]
    if not st: break
    st.sort(key=lambda b:(b[1],b[0]))
    ferry_free(c,st[0],tdrops[di%len(tdrops)]); di+=1
print('top fed mu',mu(c),'s2',s2(np.asarray(c.frame())))
# Phase 2: feed bottom boxes one-at-a-time to rotating handoffs (decongest)
bdrops=[(13,9),(12,8),(11,9),(13,8),(12,9)]; bi=0
for _ in range(6):
    if c.levels_completed>7 or c.terminal(): break
    bb=[b for b in boxes(c) if b[0]>9 and b not in botg]
    if not bb: break
    # pick box closest to avatar
    av=_grid_scan(c)[0]
    bb.sort(key=lambda b:abs(b[0]-av[0])+abs(b[1]-av[1]))
    ok=ferry_free(c,bb[0],bdrops[bi%len(bdrops)]); bi+=1
    print('bot fed %s ok=%s mu=%d s2=%s'%(bb[0],ok,mu(c),s2(np.asarray(c.frame()))))
    # brief idle to let courier seat (decongest)
    for _ in range(3):
        if c.levels_completed>7 or c.terminal(): break
        c.step(5)
print('bottom fed mu',mu(c))
mn=(99,99)
while c.levels_completed==7 and not c.terminal():
    c.step(5)
    ss=s2(np.asarray(c.frame())); 
    if ss[0]+ss[1]<mn[0]+mn[1]: mn=ss
print('FINAL lvl',c.levels_completed,'mu',mu(c),'s2',s2(np.asarray(c.frame())),'min_s2',mn)
