import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def av(env):
    bl=[b for b in P.connected_components(env.frame(),colors=[9]) if b.area==24 and (b.bbox[2]-b.bbox[0])==4 and (b.bbox[3]-b.bbox[1])==4]
    return bl[0].bbox[:2] if bl else None
env=A.Arena('g50t')
pre=[4,4,4,4,5]
for a in pre: env.step(a)
start=av(env); q=deque([(env.clone(),start,[])]); seen={start}; paths={start:[]}
while q:
    node,pos,path=q.popleft()
    for act in (1,2,3,4):
        c=node.clone(); c.step(act); p=av(c)
        if p is None or p==pos or p in seen: continue
        seen.add(p); paths[p]=path+[act]; q.append((c,p,path+[act]))
tgt=(50,50)
full=pre+paths[tgt]
print("FULL PATH",full,"len",len(full))
# validate on fresh env
e=A.Arena('g50t')
for a in full:
    e.step(a)
    if e.levels_completed>0: break
print("levels after path:",e.levels_completed,"steps taken:",len(e.path))
