import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from collections import deque
def prog(env):
    start=env.clone()
    def k(e): return np.asarray(e.frame()).tobytes()
    seen={k(start)}; q=deque([(start,[])])
    termdepths=[]; earliest=None
    while q:
        node,pth=q.popleft()
        for a in (1,2,3,4,5):
            c=node.clone(); c.step(a)
            key=k(c)
            if c.terminal():
                termdepths.append((len(pth)+1, c.levels_completed))
                if earliest is None or len(pth)+1<earliest[0]:
                    earliest=(len(pth)+1, c.levels_completed, pth+[a])
                continue
            if key in seen: continue
            seen.add(key); q.append((c,pth+[a]))
    print("terminal depths (depth,lvl):", sorted(set(termdepths)))
    print("earliest terminal:", earliest[:2], "path", earliest[2])
    raise SystemExit
A.run_program('g50t', prog)
