import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
env=A.Arena('g50t')
key=lambda e: e.frame().tobytes()
q=deque([env.clone()]); seen={key(env)}
win=False
while q and len(seen)<6000:
    node=q.popleft()
    if node.levels_completed>0: win=True; break
    for act in (1,2,3,4,5):
        c=node.clone(); c.step(act); k=key(c)
        if k in seen: continue
        seen.add(k); q.append(c)
print("closure states",len(seen),"win",win,"queue_empty",len(q)==0)
