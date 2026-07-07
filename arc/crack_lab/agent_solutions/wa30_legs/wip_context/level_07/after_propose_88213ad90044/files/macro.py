from probe6 import fresh_at_level6
from sim6 import parse, deliver
import numpy as np, time
from collections import deque

PREFIX=[1]*7+[4]*6+[5]
CAND=[(3,7),(3,8),(4,7),(4,8),(6,13),(6,14),(7,13),(7,14),
      (6,8),(9,14),(3,14),(4,14)]

def base_clone():
    b=fresh_at_level6(budget=200_000_000).clone()
    for a in PREFIX: b.step(a)
    return b

def boxset(c):
    return frozenset(parse(np.asarray(c.frame()))[1])

def macro_bfs(time_cap=200, max_states=4000):
    t0=time.time()
    root=base_clone()
    s0=boxset(root)
    # store representative clone per state
    q=deque([(s0,[])]); seen={s0}
    while q and time.time()-t0<time_cap and len(seen)<max_states:
        state,plan=q.popleft()
        for src in state:
            for dst in CAND:
                if dst in state and dst!=src: continue
                if dst==src: continue
                c=base_clone()
                ok=True
                # replay plan
                for (b,d) in plan:
                    if not deliver(c,b,d): ok=False;break
                if not ok: continue
                # ensure src still a box
                cur=boxset(c)
                if src not in cur: continue
                if not deliver(c,src,dst): continue
                if c.levels_completed>5:
                    return plan+[(src,dst)], c
                ns=boxset(c)
                if ns==state: continue
                if ns in seen: continue
                seen.add(ns)
                q.append((ns,plan+[(src,dst)]))
    return None, len(seen)

if __name__=='__main__':
    res,info=macro_bfs()
    print('result',res,'info',info)
