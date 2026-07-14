import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque

def av(env):
    bl=[b for b in P.connected_components(env.frame(),colors=[9]) if b.area==24 and (b.bbox[2]-b.bbox[0])==4 and (b.bbox[3]-b.bbox[1])==4]
    return bl[0].bbox[:2] if bl else None

env=A.Arena('g50t')
start=env.clone()
key=lambda e: e.frame().tobytes()
q=deque([(start,[])]); seen={key(start)}
best=None
found=None
maxrow=8
while q and len(seen)<12000:
    node,path=q.popleft()
    if node.levels_completed>0:
        found=path; break
    p=av(node)
    if p and p[0]>maxrow:
        maxrow=p[0]; best=(p,path)
    if len(path)>=45: continue
    for act in (1,2,3,4,5):
        c=node.clone(); c.step(act)
        k=key(c)
        if k in seen: continue
        seen.add(k); q.append((c,path+[act]))
print("found reward path:",found)
print("deepest avatar row:",maxrow,"via",best[1] if best else None)
print("states",len(seen))
