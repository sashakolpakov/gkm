from l8env import l8
from ferry import ferry_free
from legs import _grid_scan
import numpy as np
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
def goalfill(f):
    return int((f[9:15,45:59]==2).sum())+int((f[49:59,49:59]==2).sum())
c=l8().clone()
topdrops=[(4,7),(3,7)]; botdrops=[(13,9),(12,9)]
di=0
def stuck_top(c): return [b for b in _grid_scan(c)[1] if b[0]<6 and b[1]<5]
def stuck_bot(c): return [b for b in _grid_scan(c)[1] if b[0]>9 and b[1]<6]
def idle(c,n):
    for _ in range(n):
        if c.terminal() or c.levels_completed>7: return
        c.step(5)
for rnd in range(8):
    if c.levels_completed>7 or c.terminal(): break
    st=stuck_top(c)
    if st:
        st.sort(key=lambda b:(b[1],b[0]))
        ok=ferry_free(c,st[0],topdrops[di%2]); di+=1
        print('top ferry',st[0],ok,'mu',mu(c),'gf',goalfill(np.asarray(c.frame())),'lvl',c.levels_completed)
        idle(c,4)
    sb=stuck_bot(c)
    if sb:
        sb.sort(key=lambda b:(b[1],b[0]))
        ok=ferry_free(c,sb[0],botdrops[di%2]); di+=1
        print('bot ferry',sb[0],ok,'mu',mu(c),'gf',goalfill(np.asarray(c.frame())),'lvl',c.levels_completed)
        idle(c,4)
    if not st and not sb: break
print('ferry done mu',mu(c))
idle(c,60)
f=np.asarray(c.frame())
print('FINAL lvl',c.levels_completed,'mu',mu(c),'goalfill',goalfill(f))
print('remaining floor boxes',sorted(_grid_scan(c)[1]))
