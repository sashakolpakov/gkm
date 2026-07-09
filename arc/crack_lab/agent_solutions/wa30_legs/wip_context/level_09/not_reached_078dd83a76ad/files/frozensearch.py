import probe7, legs
import numpy as np
from collections import deque

env=probe7.get_env_at_L7()
def cour(e):
    f=np.asarray(e.frame()); ys,xs=np.where(f==15); return (int(ys.min())//4,int(xs.min())//4)
c=env.clone(); prev=None; stable=0
for i in range(40):
    p=cour(c); stable=stable+1 if p==prev else 0; prev=p
    if stable>=5 and i>16: break
    c.step(legs.UP)
frozen=c; base=frozen.levels_completed
print('frozen lc',base,'courier',cour(frozen),'boxes',sorted(legs._grid_scan(frozen)[1]))

def key(e):
    f=np.asarray(e.frame()); parts=[]
    for R in range(5,10):
        for C in range(16):
            u=frozenset(int(v) for v in np.unique(f[R*4:R*4+4,C*4:C*4+4]) if v in (0,2,3,4,5,9,14,15))
            if u: parts.append((R,C,u))
    return tuple(parts)

q=deque([(frozen.clone(),[])]); seen={key(frozen)}; found=None; exp=0
while q:
    node,path=q.popleft(); exp+=1
    if exp>40000: break
    if len(path)>=34: continue
    for a in (1,2,3,4,5):
        ch=node.clone(); ch.step(a)
        if ch.levels_completed>base: found=path+[a]; break
        if ch.terminal(): continue
        k=key(ch)
        if k in seen: continue
        seen.add(k); q.append((ch,path+[a]))
    if found: break
print('exp',exp,'FOUND',found)
