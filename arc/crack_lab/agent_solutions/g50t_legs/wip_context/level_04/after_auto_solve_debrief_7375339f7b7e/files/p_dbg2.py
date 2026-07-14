import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def av(env):
    bl=[b for b in P.connected_components(env.frame(),colors=[9]) if b.area==24 and (b.bbox[2]-b.bbox[0])==4 and (b.bbox[3]-b.bbox[1])==4]
    return bl[0].bbox[:2] if bl else None
def leg(e): return hash(e.frame()[0:7,0:12].tobytes())%1000
def key(e): return (av(e), e.frame()[0:7,0:12].tobytes(), int(e.levels_completed))
env=A.Arena('g50t')
q=deque([(env.clone(),[])]); seen={key(env)}; states=[]
while q and len(seen)<40000:
    n,p=q.popleft()
    if len(p)>=120: continue
    for a in (1,2,3,4,5):
        c=n.clone(); c.step(a)
        if c.levels_completed>0: print("WIN",p+[a]); sys.exit()
        k=key(c)
        if k in seen: continue
        seen.add(k); states.append((av(c),leg(c),p+[a]))
        q.append((c,p+[a]))
for pos,lg,p in states: print(pos,lg,"path",p)
