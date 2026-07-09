import probe7, legs
import numpy as np
from collections import deque

env=probe7.get_env_at_L7()

def key(e):
    f=np.asarray(e.frame())
    # coarse cell signature for relevant colors in play area rows 20-39
    parts=[]
    for R in range(5,10):
        for C in range(0,16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            u=frozenset(int(v) for v in np.unique(blk) if v in (0,2,3,4,9,14,15))
            if u: parts.append((R,C,u))
    return tuple(parts)

start=env.clone()
base=start.levels_completed
q=deque([(start,[])])
seen={key(start)}
found=None
exp=0
while q:
    node,path=q.popleft()
    exp+=1
    if exp>6000: break
    if len(path)>=28: continue
    for a in (1,2,3,4,5):
        ch=node.clone(); ch.step(a)
        if ch.levels_completed>base:
            found=path+[a]; break
        if ch.terminal(): continue
        k=key(ch)
        if k in seen: continue
        seen.add(k)
        q.append((ch,path+[a]))
    if found: break
print('expansions',exp,'found',found)
