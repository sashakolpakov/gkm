import l9env, legs
import numpy as np
from collections import deque
_DIRS={1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
def bfs(start,goal,blocked,G=16):
    if start==goal: return []
    q=deque([start]); prev={start:None}
    while q:
        cur=q.popleft()
        for a,(dr,dc) in _DIRS.items():
            n=(cur[0]+dr,cur[1]+dc)
            if not(0<=n[0]<G and 0<=n[1]<G):continue
            if n in prev:continue
            if n in blocked and n!=goal:continue
            prev[n]=(cur,a)
            if n==goal:
                p=[];x=n
                while prev[x] is not None:pp,ac=prev[x];p.append(ac);x=pp
                return p[::-1]
            q.append(n)
    return None
def full2cells(f,G=16):
    out=[]
    for R in range(G):
        for C in range(G):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            if np.all(blk==2): out.append((R,C))
    return out
env=l9env.get_l9()
f=np.asarray(env.frame())
print("full-2 cells (solid 2x2 payload/target):",full2cells(f))
av,boxes,walls=legs._grid_scan(env)
blocked=set(walls)|set(boxes)
path=bfs(av,(8,3),blocked)
for a in path: env.step(a)
c2before=int((np.asarray(env.frame())==2).sum())
env.step(3) # move left onto 2222 (8,2)
env.step(4) # move right away
f=np.asarray(env.frame())
print("after step onto and off (8,2): full-2 cells now:",full2cells(f),"c2",int((f==2).sum()),"was",c2before,"lvl",env.levels_completed)
for r in range(32,36):
    print(f'{r}: '+''.join(f'{int(f[r,c]):x}' if f[r,c] else '.' for c in range(4,20)))
