import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
# From EVERY reachable state, record c8 and the exact 8-cell set; list all distinct 8-configs.
def prog(env):
    start=env.clone()
    def mk(e):
        f=np.asarray(e.frame()).copy(); f[63,:]=0; return f.tobytes()
    seen={mk(start)}; q=deque([start]); configs8=set()
    while q:
        n=q.popleft()
        f=np.asarray(n.frame())
        ys,xs=np.where(f==8)
        configs8.add(frozenset(zip(ys.tolist(),xs.tolist())))
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a)
            if c.terminal(): continue
            k=mk(c)
            if k in seen: continue
            seen.add(k); q.append(c)
    print("distinct 8-configs:",len(configs8))
    for cfg in configs8:
        ys=[p[0] for p in cfg]; xs=[p[1] for p in cfg]
        print("  count",len(cfg),"rows",min(ys),max(ys),"cols",min(xs),max(xs))
    raise SystemExit
A.run_program('g50t', prog)
