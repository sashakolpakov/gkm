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
                while prev[x] is not None:
                    pp,ac=prev[x];p.append(ac);x=pp
                return p[::-1]
            q.append(n)
    return None
for goal in [(15,0),(15,1),(14,0),(15,7),(15,5)]:
    env=l9env.get_l9()
    av,head,boxes,cour,walls=legs._cells(env)
    blocked=set(walls)|set(boxes)
    path=bfs(av,goal,blocked)
    if path is None:
        print(goal,"NO PATH"); continue
    for a in path:
        if env.terminal():break
        env.step(a)
    print(goal,"pathlen",len(path),"-> lvl",env.levels_completed,"term",env.terminal(),"avcell",legs._cells(env)[0])
