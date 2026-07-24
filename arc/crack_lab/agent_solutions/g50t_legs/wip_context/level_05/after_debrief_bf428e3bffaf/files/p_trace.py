import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
from collections import deque
env=A.Arena('g50t'); key=lambda e:e.frame().tobytes()
q=deque([env.clone()]); seen={key(env)}
while q:
    n=q.popleft()
    for a in (1,2,3,4,5):
        c=n.clone(); c.step(a); k=key(c)
        if k in seen: continue
        seen.add(k); q.append(c)
path=[4,4,4,4,5, 2,2,2,2,2,2,2, 4,4,4,4,4]
e=A.Arena('g50t')
for i,a in enumerate(path):
    e.step(a)
    print(i+1,"act",a,"inClosure",key(e) in seen,"lvl",e.levels_completed)
