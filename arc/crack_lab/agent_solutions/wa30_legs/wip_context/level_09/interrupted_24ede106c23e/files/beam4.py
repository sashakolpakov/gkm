from l8env import l8
from legs import _grid_scan
import numpy as np, time
def frame(c): return np.asarray(c.frame())
def movers(f): return frozenset((int(y)//4,int(x)//4) for y,x in zip(*np.where(f==15)))
def c12(f): return frozenset((int(y)//4,int(x)//4) for y,x in zip(*np.where(f==12)))
def total2(f): return int((f==2).sum())
def key(c):
    f=frame(c); av,boxes,walls=_grid_scan(c)
    return (av,movers(f),c12(f),frozenset(boxes))
def score(c):
    if c.levels_completed>7: return 1e9
    return -total2(frame(c))
start=l8()
beam=[(score(start),start.clone(),[])]; seen={key(start)}; best=(-9999,[]); found=None; t0=time.time()
mn=9999
for d in range(134):
    cand=[]
    for sc,c,path in beam:
        for a in (1,2,3,4,5):
            n=c.clone()
            if n.terminal(): continue
            n.step(a)
            if n.levels_completed>7: found=path+[a]; break
            k=key(n)
            if k in seen: continue
            seen.add(k); s=score(n); cand.append((s,n,path+[a])); mn=min(mn,total2(frame(n)))
        if found: break
    if found: break
    if not cand: break
    cand.sort(key=lambda x:-x[0]); beam=cand[:120]
    if beam[0][0]>best[0]: best=beam[0]
    if d%20==0: print('d%d best_total2=%d min_seen=%d el=%.0f'%(d,-best[0],mn,time.time()-t0))
print('found',found is not None,'best_total2',-best[0],'min_total2',mn)
