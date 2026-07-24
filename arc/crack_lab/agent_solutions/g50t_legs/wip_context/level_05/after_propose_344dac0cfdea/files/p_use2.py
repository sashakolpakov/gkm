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
seen={start:env.clone()}
q=deque([(env.clone(),start)])
while q:
    node,pos=q.popleft()
    for act in (1,2,3,4):
        c=node.clone(); c.step(act)
        np_=avatar_tl(c)
        if np_ is None or np_==pos or np_ in seen: continue
        seen[np_]=c; q.append((c,np_))

for pos,cl in sorted(seen.items()):
    before=cl.frame()
    c2=cl.clone(); c2.step(5)
    diff=(c2.frame()!=before).sum()
    cc0=P.color_counts(before); cc1=P.color_counts(c2.frame())
    print(pos,"USEdiff",int(diff),"d8",cc1.get(8,0)-cc0.get(8,0),"d9",cc1.get(9,0)-cc0.get(9,0),"lvl",c2.levels_completed)
