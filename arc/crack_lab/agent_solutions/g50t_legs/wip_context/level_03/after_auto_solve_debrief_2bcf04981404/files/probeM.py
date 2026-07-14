import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def key(env): return np.asarray(env.frame()).tobytes()
def prog(env):
    start=env.clone(); seen={key(start):[]}; q=deque([(start,[])])
    minc8=999; best=None
    goalav=None
    c9set=set()
    while q:
        n,pth=q.popleft()
        cc=P.color_counts(n.frame())
        c8=cc.get(8,0)
        if c8<minc8: minc8=c8; best=(pth,cc)
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a); k=key(c)
            if k in seen: continue
            seen[k]=pth+[a]; q.append((c,pth+[a]))
    print("min c8",minc8,"at path len",len(best[0]),"counts",best[1])
    print("path to min c8:",best[0])
    raise SystemExit
A.run_program('g50t', prog)
