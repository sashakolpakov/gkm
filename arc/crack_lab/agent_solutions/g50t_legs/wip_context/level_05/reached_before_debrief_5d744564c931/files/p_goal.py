import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
from collections import deque
def av(env):
    bl=[b for b in P.connected_components(env.frame(),colors=[9]) if b.area==24 and (b.bbox[2]-b.bbox[0])==4 and (b.bbox[3]-b.bbox[1])==4]
    return bl[0].bbox[:2] if bl else None
env=A.Arena('g50t')
for a in [4,4,4,4,5]: env.step(a)  # open gates
# BFS movement to each target position, record path
start=av(env); q=deque([(env.clone(),start,[])]); seen={start}; paths={start:[]}
while q:
    node,pos,path=q.popleft()
    for act in (1,2,3,4):
        c=node.clone(); c.step(act); p=av(c)
        if p is None or p==pos or p in seen: continue
        seen.add(p); paths[p]=path+[act]; q.append((c,p,path+[act]))
for tgt in [(50,44),(50,38),(44,50),(50,50)]:
    if tgt in paths:
        c=env.clone()
        for a in paths[tgt]: c.step(a)
        # try use
        cu=c.clone(); cu.step(5)
        print(tgt,"reached lvl",c.levels_completed,"| afterUSE lvl",cu.levels_completed,"path",len(paths[tgt]))
