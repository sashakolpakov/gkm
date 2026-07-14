import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque

def key(env):
    return np.asarray(env.frame()).tobytes()
def av(env):
    for b in P.connected_components(env.frame(),colors=[9]):
        if b.area==24: return b.bbox[:2]
    return None

def prog(env):
    start=env.clone(); seen={key(start)}; q=deque([start])
    avset=set(); legendset=set(); mazeset=set()
    while q and len(seen)<20000:
        n=q.popleft()
        f=np.asarray(n.frame())
        avset.add(av(n))
        legendset.add(f[0:6,0:9].tobytes())
        mazeset.add(f[7:,10:].tobytes())
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a); k=key(c)
            if k not in seen: seen.add(k); q.append(c)
    print("states",len(seen),"avatar positions",len(avset),"legend variants",len(legendset),"maze variants",len(mazeset))
    print("avatar list",sorted(x for x in avset if x))
    # print a couple legend variants
    raise SystemExit
A.run_program('g50t', prog)
