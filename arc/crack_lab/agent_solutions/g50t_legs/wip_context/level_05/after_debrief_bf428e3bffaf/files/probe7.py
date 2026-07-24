import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque

def av(env):
    for b in P.connected_components(env.frame(),colors=[9]):
        if b.area==24:
            return b.bbox[:2]
    # if merged, find by exclusion? return None
    return None

def prog(env):
    start=env.clone()
    s0=av(start)
    print("start avatar tl",s0)
    seen={s0:[]}
    q=deque([(start,s0,[])])
    reached=set([s0])
    steps=0
    while q and steps<3000:
        node,pos,pth=q.popleft()
        for a in (1,2,3,4):
            c=node.clone(); c.step(a)
            p=av(c)
            if p is None:
                # avatar merged/covered goal -> record special
                print("avatar None after path",pth+[a],"level",c.levels_completed)
                continue
            if p in reached: continue
            reached.add(p); seen[p]=pth+[a]
            q.append((c,p,pth+[a]))
        steps+=1
    print("num reachable",len(reached))
    rs=[p[0] for p in reached]; cs=[p[1] for p in reached]
    print("row range",min(rs),max(rs),"col range",min(cs),max(cs))
    # grid map
    for r in sorted(set(rs)):
        line=''.join('#' if (r,c) in reached else '.' for c in range(0,60))
        print(f"{r:2d} {line}")
    raise SystemExit
A.run_program('g50t', prog)
