from l8env import l8
from legs import _grid_scan
import numpy as np
from collections import deque
def movers(f,col): return tuple(sorted((int(y)//4,int(x)//4) for y,x in zip(*np.where(f==col))))
def key(c):
    f=np.asarray(c.frame()); av,boxes,walls=_grid_scan(c)
    return (av, movers(f,15), movers(f,12), frozenset(boxes))
start=l8()
q=deque([(start.clone(),[])]); seen={key(start)}
import time; t0=time.time(); best=0
while q:
    c,path=q.popleft()
    if len(path)>=45: continue
    for a in (1,2,3,4,5):
        n=c.clone()
        if n.terminal(): continue
        n.step(a)
        if n.levels_completed>7:
            print('WIN len',len(path)+1,'path',path+[a]); 
            import json; json.dump(path+[a],open('l8win.json','w')); raise SystemExit
        k=key(n)
        if k in seen: continue
        seen.add(k); q.append((n,path+[a]))
    if len(seen)%20000<6:
        print('seen',len(seen),'depth~',len(path),'el',round(time.time()-t0))
    if len(seen)>250000 or time.time()-t0>150: print('stop, seen',len(seen)); break
print('no win found in BFS')
