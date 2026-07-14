import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from collections import deque
# reference frames along winning path
path=[4,4,4,4,5, 2,2,2,2,2,2,2, 4,4,4,4,4]
e=A.Arena('g50t'); frames=[e.frame().tobytes()]
for a in path:
    e.step(a); frames.append(e.frame().tobytes())
f_gatesopen=frames[5]   # after [4,4,4,4,5]
f_step16=frames[16]
# full BFS closure record
env=A.Arena('g50t')
key=lambda e:e.frame().tobytes()
q=deque([env.clone()]); seen={key(env)}
while q:
    n=q.popleft()
    for a in (1,2,3,4,5):
        c=n.clone(); c.step(a); k=key(c)
        if k in seen: continue
        seen.add(k); q.append(c)
print("total seen",len(seen))
print("gates-open frame in seen:",f_gatesopen in seen)
print("step16 frame in seen:",f_step16 in seen)
