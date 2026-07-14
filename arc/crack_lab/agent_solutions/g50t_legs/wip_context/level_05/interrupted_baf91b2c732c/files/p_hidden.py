import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
def fr(path):
    e=A.Arena('g50t')
    for a in path: e.step(a)
    return e
e1=fr([4,4,4,4,5])      # USE at (8,38)
e2=fr([2,5])            # USE at (14,14)
e0=fr([])
f1,f2,f0=e1.frame(),e2.frame(),e0.frame()
print("f1==f2:",(f1==f2).all(), "f1==f0:",(f1==f0).all(),"f2==f0:",(f2==f0).all())
# reachability from each (moves only)
import perception as P
from collections import deque
def av(env):
    bl=[b for b in P.connected_components(env.frame(),colors=[9]) if b.area==24 and (b.bbox[2]-b.bbox[0])==4 and (b.bbox[3]-b.bbox[1])==4]
    return bl[0].bbox[:2] if bl else None
def reach(env):
    s=av(env); seen={s}; q=deque([(env.clone(),s)])
    while q:
        n,pos=q.popleft()
        for a in (1,2,3,4):
            c=n.clone(); c.step(a); p=av(c)
            if p is None or p==pos or p in seen: continue
            seen.add(p); q.append((c,p))
    return len(seen)
print("reach e1(USE@8,38):",reach(e1))
print("reach e2(USE@14,14):",reach(e2))
print("reach e0(start):",reach(e0))
