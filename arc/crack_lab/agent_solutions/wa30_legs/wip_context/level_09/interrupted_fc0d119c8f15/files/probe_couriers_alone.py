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
def ci2(e):
    f=np.asarray(e.frame()); return int((f[13:23,21:31]==2).sum())
env=l9env.get_l9()
av,boxes,walls=legs._grid_scan(env)
blocked=set(walls)|set(boxes)
# park top-left corner (0,0)
p=bfs(av,(0,0),blocked)
for a in p: env.step(a)
print("parked, steps",len(env.path)-588,"ci2",ci2(env))
for t in range(40):
    if env.terminal():break
    env.step(3) # idle LEFT (already at corner, won't move)
    if t%5==0:
        print(f"t{t} ci2{ci2(env)} lvl{env.levels_completed} n7{int((np.asarray(env.frame())==7).sum())}")
print("final ci2",ci2(env),"lvl",env.levels_completed)
