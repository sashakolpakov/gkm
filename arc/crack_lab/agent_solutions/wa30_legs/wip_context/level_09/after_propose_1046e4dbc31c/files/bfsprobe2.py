from l8env import l8
from legs import _grid_scan
import numpy as np
from collections import deque
import time,json
def mv(f,col): return tuple(sorted((int(y)//4,int(x)//4) for y,x in zip(*np.where(f==col))))
def key(c):
    f=np.asarray(c.frame()); av,boxes,walls=_grid_scan(c)
    return (av, mv(f,15), mv(f,12), frozenset(boxes))
start=l8()
q=deque([(start.clone(),0)]); seen={key(start):None}
# store parent for path reconstruction
parent={key(start):(None,None)}
t0=time.time(); maxd=0
win=None
while q:
    c,d=q.popleft()
    maxd=max(maxd,d)
    if d>=120: continue
    ck=key(c)
    for a in (1,2,3,4,5):
        n=c.clone()
        if n.terminal(): continue
        n.step(a)
        if n.levels_completed>7:
            win=(ck,a); print('WIN at depth',d+1); break
        k=key(n)
        if k in seen: continue
        seen[k]=None; parent[k]=(ck,a); q.append((n,d+1))
    if win: break
    if len(seen)>800000 or time.time()-t0>220:
        print('cap hit seen',len(seen),'maxd',maxd,'time',round(time.time()-t0)); break
print('total seen',len(seen),'maxdepth',maxd,'win',win is not None)
if win:
    # reconstruct
    ck,a=win; path=[a]
    while ck is not None:
        p,pa=parent[ck]
        if pa is None: break
        path.append(pa); ck=p
    path=path[::-1]
    print('winpath len',len(path)); json.dump(path,open('l8win.json','w'))
