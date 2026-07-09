from l8env import l8
from legs import grab_and_deliver, _grid_scan, _cells
import numpy as np
def cc(f,R,C): return set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
def gf(f): return int((f[9:15,45:59]==2).sum())+int((f[49:59,49:59]==2).sum())
topg=set((r,co) for r in (2,3) for co in (11,12,13,14))
botg=set((r,co) for r in (12,13,14) for co in (12,13,14))
goals=topg|botg
top_h=[(2,10),(3,10),(1,9)]; bot_h=[(13,11),(12,11),(11,10)]
c=l8().clone()
cnt=[0]; o=c.step
def cs(a):
    if cnt[0]>=134: raise RuntimeError('cap')
    cnt[0]+=1; return o(a)
c.step=cs
ti=0; bi=0
try:
  while cnt[0]<134 and c.levels_completed==7:
    f=np.asarray(c.frame())
    av,boxes,walls=_grid_scan(c)
    free=[b for b in boxes if b not in goals]
    if not free:
        c.step(5); continue
    free.sort(key=lambda b:abs(b[0]-av[0])+abs(b[1]-av[1]))
    box=free[0]
    if box[0]<6:
        h=top_h[ti%len(top_h)]; ti+=1
    else:
        h=bot_h[bi%len(bot_h)]; bi+=1
    ok=grab_and_deliver(c,box,h)
    if cnt[0]%20<3 or c.levels_completed>7:
        print('box%s->%s ok=%s steps=%d gf=%d lvl=%d'%(box,h,ok,cnt[0],gf(np.asarray(c.frame())),c.levels_completed))
    if not ok:
        c.step(5)  # nudge
except RuntimeError: pass
print('FINAL lvl',c.levels_completed,'steps',cnt[0],'gf',gf(np.asarray(c.frame())))
