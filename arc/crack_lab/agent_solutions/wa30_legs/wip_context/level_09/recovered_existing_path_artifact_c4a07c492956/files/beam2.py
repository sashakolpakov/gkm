from l8env import l8
from legs import _grid_scan
import numpy as np, time
def frame(c): return np.asarray(c.frame())
def cc(f,R,C): return set(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]))
reg=[(2,3),(3,3),(4,1),(4,2),(4,3),(12,3),(12,4),(12,5),(13,3),(13,4),(13,5),(14,3),(14,4),(14,5)]
def covered(f): return sum(1 for (R,C) in reg if {4,3,0}&cc(f,R,C))
def movers(f):
    ys,xs=np.where(f==15); return frozenset((int(y)//4,int(x)//4) for y,x in zip(ys,xs))
def key(c):
    f=frame(c); av,boxes,walls=_grid_scan(c)
    return (av,movers(f),frozenset(boxes))
def score(c):
    if c.levels_completed>7: return 1e9
    return covered(frame(c))
start=l8()
beam=[(score(start),start.clone(),[])]; seen={key(start)}; best=(0,[]); found=None; t0=time.time()
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
            seen.add(k); cand.append((score(n),n,path+[a]))
        if found: break
    if found: break
    if not cand: break
    cand.sort(key=lambda x:-x[0]); beam=cand[:60]
    if beam[0][0]>best[0]: best=(beam[0][0],beam[0][2])
    if d%15==0: print('d%d bestcov=%d elapsed=%.0f'%(d,best[0],time.time()-t0))
print('found',found is not None,'bestcov',best[0],'of',len(reg))
