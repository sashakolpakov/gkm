from l8env import l8
from legs import _grid_scan
import numpy as np, sys, time
def frame(c): return np.asarray(c.frame())
def interior2(f):
    t=int((f[9:15,45:59]==2).sum())
    b=int((f[49:59,49:59]==2).sum())
    return t,b
def free_boxes(c):
    av,boxes,walls=_grid_scan(c)
    fb=[b for b in boxes if not(2<=b[0]<=3 and b[1]>=11) and not(12<=b[0]<=14 and b[1]>=12)]
    return av,fb,boxes
def movers(f):
    ys,xs=np.where(f==15); return frozenset((int(y)//4,int(x)//4) for y,x in zip(ys,xs))
def score(c):
    f=frame(c)
    if c.levels_completed>7: return 1e9
    t,b=interior2(f)
    av,fb,boxes=free_boxes(c)
    return -(t+b) - 6*len(fb)
def key(c):
    f=frame(c)
    av,fb,boxes=free_boxes(c)
    return (av,movers(f),frozenset(boxes))

start=l8()
beam=[(score(start),start.clone(),[])]
seen=set([key(start)])
best=(score(start),[])
t0=time.time()
DEPTH=130; W=60
found=None
for d in range(DEPTH):
    cand=[]
    for sc,c,path in beam:
        for a in (1,2,3,4,5):
            n=c.clone()
            if n.terminal(): continue
            n.step(a)
            if n.levels_completed>7:
                found=path+[a]; print('FOUND lvl up at depth',d,'len',len(found)); break
            k=key(n)
            if k in seen: continue
            seen.add(k)
            s=score(n)
            cand.append((s,n,path+[a]))
        if found: break
    if found: break
    if not cand: break
    cand.sort(key=lambda x:-x[0])
    beam=cand[:W]
    if beam[0][0]>best[0]: best=(beam[0][0],beam[0][2])
    if d%10==0: print('d%d best_sc=%.0f beamtop=%.0f elapsed=%.0f'%(d,best[0],beam[0][0],time.time()-t0))
print('found',found is not None,'best_sc',best[0])
if found:
    import json; json.dump(found, open('l8path.json','w')); print('saved',len(found))
