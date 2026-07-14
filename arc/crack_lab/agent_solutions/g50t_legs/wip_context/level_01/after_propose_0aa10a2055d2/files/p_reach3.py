import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def av(env):
    bl=[b for b in P.connected_components(env.frame(),colors=[9]) if b.area==24 and (b.bbox[2]-b.bbox[0])==4 and (b.bbox[3]-b.bbox[1])==4]
    return bl[0].bbox[:2] if bl else None
def reachset(env):
    s=av(env); seen={s}; q=deque([(env.clone(),s)])
    while q:
        node,pos=q.popleft()
        for act in (1,2,3,4):
            c=node.clone(); c.step(act); p=av(c)
            if p is None or p==pos or p in seen: continue
            seen.add(p); q.append((c,p))
    return seen
env=A.Arena('g50t')
# toggle once
for a in [4,4,4,4,5]: env.step(a)
print("after 1 toggle, reach:",sorted(reachset(env)))
