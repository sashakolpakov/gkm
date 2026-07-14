import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from collections import deque
env=A.Arena('g50t')
key=lambda e: e.frame().tobytes()
q=deque([(env.clone(),0)]); seen={key(env)}
win=None; maxlen=0
while q:
    node,dep=q.popleft()
    maxlen=max(maxlen,dep)
    if node.levels_completed>0: win=dep; break
    for act in (1,2,3,4,5):
        c=node.clone(); c.step(act); k=key(c)
        if k in seen: continue
        seen.add(k); q.append((c,dep+1))
    if len(seen)>200000: print("cap"); break
print("win at depth",win,"states",len(seen),"maxdepth",maxlen)
