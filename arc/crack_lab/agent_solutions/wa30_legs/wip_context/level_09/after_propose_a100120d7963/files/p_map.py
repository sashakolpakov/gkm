import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
path=json.load(open('/tmp/l8_minagent.json'))
# candidate color-2 cells at start
env0=l8(); f0=np.asarray(env0.frame())
cands=[]
for R in range(16):
    for C in range(16):
        blk=f0[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
        if 2 in u and 4 not in u and 15 not in u and 12 not in u:  # color-2 region cell
            cands.append((R,C))
print("num color-2 cells",len(cands))
def test(tgt):
    env=l8()
    for a in path[:20]: env.step(a)
    ok=False
    for src in [(2,2),(2,1),(3,2),(3,1)]:
        if legs.carry_box_to(env,src,tgt,cap=40): ok=True;break
    if not ok: return None
    for _ in range(3): env.step(2)
    f=np.asarray(env.frame())
    ys,xs=np.where(f==3); locs=set((int(y)//4,int(x)//4) for y,x in zip(ys,xs))
    return tgt in locs
goals=[]
for t in cands:
    r=test(t)
    if r: goals.append(t)
    print(t,r)
print("GOAL TILES:",goals,"count",len(goals))
json.dump(goals,open('/tmp/l8_goals.json','w'))
