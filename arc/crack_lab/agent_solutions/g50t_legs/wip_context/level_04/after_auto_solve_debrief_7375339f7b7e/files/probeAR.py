import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from collections import deque
def prog(env):
    # full-frame key INCLUDING counter; check every child for win
    start=env.clone()
    def k(e): return np.asarray(e.frame()).tobytes()
    seen={k(start)}; q=deque([(start,[])]); win=None; maxd=0; terms=0
    while q:
        node,pth=q.popleft(); maxd=max(maxd,len(pth))
        for a in (1,2,3,4,5):
            c=node.clone(); c.step(a)
            if c.levels_completed>0:
                win=pth+[a]; print("WIN",win); raise SystemExit
            key=k(c)
            if c.terminal(): terms+=1; continue
            if key in seen: continue
            seen.add(key); q.append((c,pth+[a]))
        if len(seen)%20000==0: print("...",len(seen),"maxd",maxd)
        if len(seen)>250000: print("CAP"); break
    print("total",len(seen),"maxdepth",maxd,"terminals",terms,"win",win)
    raise SystemExit
A.run_program('g50t', prog)
