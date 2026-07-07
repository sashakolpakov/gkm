from l8env import l8
from legs import _grid_scan
import numpy as np, time
def frame(c): return np.asarray(c.frame())
def movers(f):
    ys,xs=np.where(f==15); return frozenset((int(y)//4,int(x)//4) for y,x in zip(ys,xs))
def fb(c):
    av,boxes,walls=_grid_scan(c); return len(boxes)
def key(c):
    f=frame(c); av,boxes,walls=_grid_scan(c)
    return (av,movers(f),frozenset(boxes))
def score(c):
    if c.levels_completed>7: return 1e9
    return -fb(c)
start=l8()
beam=[(score(start),start.clone(),[])]; seen={key(start)}; best=(-99,[]); found=None; t0=time.time()
minfb=99
for d in range(130):
    cand=[]
    for sc,c,path in beam:
        for a in (1,2,3,4,5):
            n=c.clone()
            if n.terminal(): continue
            n.step(a)
            if n.levels_completed>7: found=path+[a]; break
            k=key(n)
            if k in seen: continue
            seen.add(k); s=score(n); cand.append((s,n,path+[a]))
            minfb=min(minfb,fb(n))
        if found: break
    if found: break
    if not cand: break
    cand.sort(key=lambda x:-x[0]); beam=cand[:80]
    if beam[0][0]>best[0]: best=(beam[0][0],beam[0][2])
    if d%15==0: print('d%d best_fb=%d minfb_seen=%d elapsed=%.0f'%(d,-best[0],minfb,time.time()-t0))
print('found',found is not None,'best_fb',-best[0],'minfb',minfb)
