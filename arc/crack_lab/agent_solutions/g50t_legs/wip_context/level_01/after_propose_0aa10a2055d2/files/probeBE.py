import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def prog(env):
    start=env.clone()
    def k(e): return np.asarray(e.frame()).tobytes()
    seen={k(start)}; q=deque([start]); twos=set()
    while q:
        n=q.popleft()
        f=np.asarray(n.frame())
        for b in P.connected_components(f,colors=[2]):
            if b.bbox[0]>=7: twos.add(b.bbox)
        for a in (1,2,3,4,5):
            c=n.clone(); c.step(a)
            if c.terminal(): continue
            key=k(c)
            if key in seen: continue
            seen.add(key); q.append(c)
    print("all distinct maze color-2 anchor bboxes:", sorted(twos))
    raise SystemExit
A.run_program('g50t', prog)
