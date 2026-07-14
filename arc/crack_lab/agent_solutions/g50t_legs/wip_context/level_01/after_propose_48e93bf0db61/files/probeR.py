import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def key(env):
    f=np.asarray(env.frame()).copy()
    f[63,:]=0  # mask move-counter row
    return f.tobytes()
def prog(env):
    start=env.clone()
    seen={key(start)}
    q=deque([(start,[])])
    found=None; term_states=0; maxdepth=0
    while q:
        n,pth=q.popleft()
        maxdepth=max(maxdepth,len(pth))
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a)
            if c.levels_completed>0:
                found=pth+[a]; print("WIN path len",len(found)); print(found); 
                raise SystemExit
            if c.terminal():
                term_states+=1
                continue
            k=key(c)
            if k in seen: continue
            seen.add(k); q.append((c,pth+[a]))
        if len(seen)>20000:
            print("cap",len(seen)); break
    print("done states",len(seen),"maxdepth",maxdepth,"terminals",term_states,"found",found)
    raise SystemExit
A.run_program('g50t', prog)
