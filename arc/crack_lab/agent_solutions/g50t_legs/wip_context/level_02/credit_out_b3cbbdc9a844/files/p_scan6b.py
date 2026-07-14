import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def av(env):
    bl=[b for b in P.connected_components(env.frame(),colors=[9]) if b.area==24 and (b.bbox[2]-b.bbox[0])==4 and (b.bbox[3]-b.bbox[1])==4]
    return bl[0].bbox[:2] if bl else None
env=A.Arena('g50t')
# reach all 12 positions with their env clones
s=av(env); states={s:env.clone()}; q=deque([(env.clone(),s)])
while q:
    node,pos=q.popleft()
    for act in (1,2,3,4):
        c=node.clone(); c.step(act); p=av(c)
        if p is None or p==pos or p in states: continue
        states[p]=c; q.append((c,p))
anyhit=0
for pos,cl in states.items():
    b=cl.frame()
    for x in range(0,64,3):
        for y in range(0,64,3):
            c=cl.clone(); c.step(6,x,y)
            d=(c.frame()!=b).sum()
            if d>0 or c.levels_completed>0:
                anyhit+=1
                print("pos",pos,"click",x,y,"diff",int(d),"lvl",c.levels_completed)
print("total hits",anyhit)
