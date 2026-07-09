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
base=l9env.get_l9()
av0,boxes0,walls0=legs._grid_scan(base)
blocked=set(walls0)|set(boxes0)
cands=[(15,15),(14,15),(15,11),(15,13),(0,0),(0,15),(2,10),(1,10),(2,12),
       (3,13),(3,10),(6,13),(2,13),(8,15),(5,10),(7,13),(15,3),(11,15),(10,15)]
for g in cands:
    env=base.clone()
    p=bfs(av0,g,blocked)
    if p is None:
        print(g,"NOPATH");continue
    for a in p:
        if env.terminal():break
        env.step(a)
    # try USE at goal too
    lv=env.levels_completed
    env.step(5)
    print(g,"len",len(p),"lvl_after_nav",lv,"lvl_after_use",env.levels_completed,"term",env.terminal())
