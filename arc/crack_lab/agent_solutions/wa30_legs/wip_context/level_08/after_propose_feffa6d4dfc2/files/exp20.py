from l8env import l8
from ferry import ferry_free
from legs import _grid_scan
import numpy as np
def mu(c):
    f=np.asarray(c.frame()); return 64-int((f[63]==7).sum())
def s2(f): return int((f[9:15,45:59]==2).sum()),int((f[49:59,49:59]==2).sum())
topg=set((r,co) for r in (2,3) for co in (11,12,13,14))
botg=set((r,co) for r in (12,13,14) for co in (12,13,14))
def bx(c): return _grid_scan(c)[1]
def nongoal(c,band):
    b=bx(c)
    if band=='t': return [x for x in b if x[0]<6 and x not in topg]
    return [x for x in b if x[0]>9 and x not in botg]
c=l8().clone()
th=[(3,10),(2,10),(3,9)]; bh=[(13,11),(12,11),(11,11)]; ti=bi=0
# interleave: feed nearest non-goal box to its band's goal-handoff
while c.levels_completed==7 and not c.terminal():
    av=_grid_scan(c)[0]
    tb=nongoal(c,'t'); bb=nongoal(c,'b')
    if not tb and not bb: break
    cand=[]
    if tb: cand.append(('t',min(tb,key=lambda x:abs(x[0]-av[0])+abs(x[1]-av[1]))))
    if bb: cand.append(('b',min(bb,key=lambda x:abs(x[0]-av[0])+abs(x[1]-av[1]))))
    band,box=min(cand,key=lambda kb:abs(kb[1][0]-av[0])+abs(kb[1][1]-av[1]))
    if band=='t': h=th[ti%3]; ti+=1
    else: h=bh[bi%3]; bi+=1
    ok=ferry_free(c,box,h)
    print('feed %s %s->%s ok=%s mu=%d s2=%s lvl=%d'%(band,box,h,ok,mu(c),s2(np.asarray(c.frame())),c.levels_completed))
    if not ok:
        if not c.terminal(): c.step(5)
    else:
        for _ in range(2):
            if c.terminal() or c.levels_completed>7: break
            c.step(5)
print('feed loop done mu',mu(c),'lvl',c.levels_completed)
while c.levels_completed==7 and not c.terminal(): c.step(5)
print('FINAL lvl',c.levels_completed,'mu',mu(c),'s2',s2(np.asarray(c.frame())))
