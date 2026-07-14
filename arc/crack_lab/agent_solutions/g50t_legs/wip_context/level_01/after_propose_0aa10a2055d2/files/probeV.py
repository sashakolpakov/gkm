import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def key(e):
    f=np.asarray(e.frame()).copy(); f[63,:]=0; return f.tobytes()
def prog(env):
    start=env.clone(); seen={key(start)}; q=deque([(start,[])])
    allconfigs=[(start,[])]
    while q:
        n,p=q.popleft()
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a); k=key(c)
            if k in seen: continue
            seen.add(k); q.append((c,p+[a])); allconfigs.append((c,p+[a]))
    print("total configs",len(allconfigs))
    # find config with lowest avatar (max row among 9-blobs excluding goal at row>=49 and legend row<7 and bottom row63)
    best=None
    for c,p in allconfigs:
        f=np.asarray(c.frame())
        for b in P.connected_components(f,colors=[9]):
            r0,c0,r1,c1=b.bbox
            if r0<7 or r0>=49 or r1==63: continue  # skip legend/goal/border
            if best is None or r0>best[0]:
                best=(r0,b.bbox,b.area,p,c.levels_completed)
    print("lowest avatar:",best[:3],"path",best[3],"lvl",best[4])
    # also report distinct avatar bboxes
    s=set()
    for c,p in allconfigs:
        f=np.asarray(c.frame())
        for b in P.connected_components(f,colors=[9]):
            r0,c0,r1,c1=b.bbox
            if r0<7 or r0>=49 or r1==63: continue
            s.add(b.bbox)
    print("distinct avatar bboxes",sorted(s))
    raise SystemExit
A.run_program('g50t', prog)
