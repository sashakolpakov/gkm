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
env=l9env.get_l9()
av,boxes,walls=legs._grid_scan(env)
blocked=set(walls)|set(boxes)
# go to (8,3) adjacent-right of 2222 at (8,2)
path=bfs(av,(8,3),blocked)
for a in path:
    if env.terminal():break
    env.step(a)
print("at",legs._grid_scan(env)[0])
def reg(f,tag):
    print(tag)
    for r in range(32,36):
        print(f'{r}: '+''.join(f'{int(f[r,c]):x}' if f[r,c] else '.' for c in range(4,20)))
f=np.asarray(env.frame()); reg(f,"before face/use")
env.step(3) # face left toward 2222
f=np.asarray(env.frame()); reg(f,"after LEFT (face)")
env.step(5) # USE
f=np.asarray(env.frame()); reg(f,"after USE")
print("grabbed",legs._grid_grabbed(env),"lvl",env.levels_completed)
