import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def av(env):
    bl=[b for b in P.connected_components(env.frame(),colors=[9]) if b.area==24 and (b.bbox[2]-b.bbox[0])==4 and (b.bbox[3]-b.bbox[1])==4]
    return bl[0].bbox[:2] if bl else None
def key(e):
    f=e.frame()
    return (av(e), f[0:7,0:12].tobytes(), int(e.levels_completed))
env=A.Arena('g50t'); base=0
q=deque([(env.clone(),[])]); seen={key(env)}; found=None
while q and len(seen)<40000:
    n,p=q.popleft()
    if len(p)>=120: continue
    for a in (1,2,3,4,5):
        c=n.clone(); c.step(a)
        if c.levels_completed>base: found=p+[a]; break
        k=key(c)
        if k in seen: continue
        seen.add(k); q.append((c,p+[a]))
    if found: break
print("found len",len(found) if found else None,"states",len(seen))
if found:
    e=A.Arena('g50t')
    for a in found: e.step(a)
    print("validate",e.levels_completed,"moves",len(found))
    print(found)
