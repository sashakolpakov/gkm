import probe7
import numpy as np
from collections import deque

env=probe7.get_env_at_L7()
base=env.levels_completed

def key(e):
    f=np.asarray(e.frame())
    parts=[]
    for R in range(5,10):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            u=frozenset(int(v) for v in np.unique(blk) if v in (0,2,3,4,5,9,14,15))
            if u: parts.append((R,C,u))
    return tuple(parts)

q=deque([(env.clone(),[])])
seen={key(env)}
found=None; exp=0
while q:
    node,path=q.popleft()
    exp+=1
    if exp%20000==0: print('exp',exp,'depth',len(path),'seen',len(seen))
    if exp>120000: break
    if len(path)>=50: continue
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
print('FOUND',found,'exp',exp)
