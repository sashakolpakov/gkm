import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque

def avatar_tl(env):
    bl=[b for b in P.connected_components(env.frame(),colors=[9]) if b.bbox[1]>10 and b.bbox[0]<40]
    bl.sort(key=lambda b:-b.area)
    return bl[0].bbox[:2] if bl else None

env=A.Arena('g50t')
start=avatar_tl(env)
seen={start}
q=deque([(env.clone(),start,[])])
paths={start:[]}
while q:
    node,pos,path=q.popleft()
    for act in (1,2,3,4):
        c=node.clone(); c.step(act)
        np_=avatar_tl(c)
        if np_ is None: continue
        if np_==pos: continue
        if np_ in seen: continue
        seen.add(np_); paths[np_]=path+[act]
        q.append((c,np_,path+[act]))
print("reachable count",len(seen))
for p in sorted(seen):
    print(p, "len", len(paths[p]))
