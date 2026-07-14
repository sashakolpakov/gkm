import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque

def key(env):
    return np.asarray(env.frame()).tobytes()

def prog(env):
    base=env.levels_completed
    goal=lambda e,p: e.levels_completed>base
    path=P.bounded_bfs(env,goal,actions=(1,2,3,4,5),key_fn=key,max_states=15000,max_depth=40)
    print("path",path)
    # count distinct states reachable
    start=env.clone(); seen={key(start)}; q=deque([start]); 
    while q and len(seen)<15000:
        n=q.popleft()
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a); k=key(c)
            if k not in seen:
                seen.add(k); q.append(c)
    print("distinct states",len(seen))
    raise SystemExit
A.run_program('g50t', prog)
