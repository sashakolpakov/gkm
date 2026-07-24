import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque

def av(env):
    for b in P.connected_components(env.frame(),colors=[9]):
        if b.area==24:
            return b.bbox[:2]
    return None

def prog(env):
    start=env.clone()
    s0=av(start)
    seen={s0:start.clone()}
    order=[s0]
    q=deque([(start,s0)])
    while q:
        node,pos=q.popleft()
        for a in (1,2,3,4):
            c=node.clone(); c.step(a)
            p=av(c)
            if p is None or p in seen: continue
            seen[p]=c.clone(); order.append(p)
            q.append((c,p))
    print("positions:",order)
    # test USE at each
    for p in order:
        e=seen[p]
        base=np.asarray(e.frame())
        c=e.clone(); c.step(5)
        d=P.frame_delta(base,c.frame())
        # also check 8 movement each normal move: compare 8 before/after a move
        print(p,"USE delta count",d['count'], d['bbox'] if d['count'] else "", "lvl",c.levels_completed)
    # check autonomous: do 8/others move on a plain move
    e=seen[s0]; base=np.asarray(e.frame())
    for a in (1,2,3,4):
        c=e.clone(); c.step(a)
        f=np.asarray(c.frame())
        # 8 cells
        b8=set(map(tuple,np.argwhere(base==8))); a8=set(map(tuple,np.argwhere(f==8)))
        print("move",a,"8 changed?",b8!=a8, "len",len(b8),len(a8))
    raise SystemExit
A.run_program('g50t', prog)
