import probe7, legs
import numpy as np
from collections import deque
env=probe7.get_env_at_L7()
c=env.clone()
legs.carry_box_to(c,(6,8),(8,3))
for _ in range(8): c.step(legs.USE)   # freeze courier at (8,4)
base=c.levels_completed
def key(e):
    f=np.asarray(e.frame()); parts=[]
    for R in range(5,10):
        for C in range(16):
            u=frozenset(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]) if v in (0,2,3,4,5,9,14,15))
            if u: parts.append((R,C,u))
    return tuple(parts)
q=deque([(c.clone(),[])]); seen={key(c)}; found=None; exp=0
while q:
    node,path=q.popleft(); exp+=1
    if exp>25000: break
    if len(path)>=22: continue
    for a in (1,2,3,4,5):
        ch=node.clone(); ch.step(a)
        if ch.levels_completed>base: found=path+[a]; break
        if ch.terminal(): continue
        k=key(ch)
        if k in seen: continue
        seen.add(k); q.append((ch,path+[a]))
    if found: break
print('exp',exp,'FOUND',found)
