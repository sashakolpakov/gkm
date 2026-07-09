import l9env, legs
import numpy as np
from collections import deque
_DIRS={1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
FACE={(-1,0):1,(1,0):2,(0,-1):3,(0,1):4}
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
tgt=(1,11)
for adj in [(2,11),(1,10),(1,12),(0,11)]:
    env=l9env.get_l9()
    av,boxes,walls=legs._grid_scan(env)
    blocked=set(walls)|set(boxes)
    blocked.discard(adj)
    p=bfs(av,adj,blocked)
    if p is None: print(adj,"NOPATH"); continue
    for a in p:
        if env.terminal():break
        env.step(a)
    reached=legs._grid_scan(env)[0]
    face=FACE[(tgt[0]-adj[0],tgt[1]-adj[1])]
    c12b=int((np.asarray(env.frame())==12).sum())
    env.step(face); env.step(5)
    c12a=int((np.asarray(env.frame())==12).sum())
    print(adj,"reached",reached,"c12",c12b,"->",c12a,"lvl",env.levels_completed)
