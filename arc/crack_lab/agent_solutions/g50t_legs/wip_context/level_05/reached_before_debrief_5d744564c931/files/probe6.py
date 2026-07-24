import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def key(env):
    f=np.asarray(env.frame())
    ys,xs=np.where(f==9)
    return frozenset(zip(ys.tolist(),xs.tolist()))

def prog(env):
    base=env.levels_completed
    goal=lambda e,p: e.levels_completed>base
    path=P.bounded_bfs(env,goal,actions=(1,2,3,4),key_fn=key,max_states=6000,max_depth=60)
    print("levelup path",path)
    # Also gather reachable positions of avatar and check overlap with goal
    from collections import deque
    start=env.clone()
    seen={key(start)}
    q=deque([(start,[])])
    goalcells=None
    positions=[]
    while q:
        node,pth=q.popleft()
        f=np.asarray(node.frame())
        for a in (1,2,3,4):
            c=node.clone(); c.step(a)
            k=key(c)
            if k in seen: continue
            seen.add(k); q.append((c,pth+[a]))
        if len(seen)>4000: break
    print("reachable states",len(seen))
    raise SystemExit
A.run_program('g50t', prog)
