import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
from collections import deque
DIRS={1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
def scan(f):
    walls=set(); boxes=set(); cont=set(); movers=[]; cours=[]; av=None
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
            if 14 in u: av=(R,C)
            if 15 in u: movers.append((R,C))
            if 12 in u: cours.append((R,C))
            if int((blk==5).sum())>=8: walls.add((R,C))
            if 9 in u and 4 in u and 2 not in u: boxes.add((R,C))
            elif 9 in u or (2 in u and 4 not in u): cont.add((R,C))  # container/fence
    return av,walls,boxes,cont,movers,cours
def bfs(start,goals,blocked):
    q=deque([start]); prev={start:None}
    while q:
        cur=q.popleft()
        if cur in goals: 
            path=[]; n=cur
            while prev[n] is not None:
                p,a=prev[n]; path.append(a); n=p
            return path[::-1]
        for a,(dr,dc) in DIRS.items():
            nb=(cur[0]+dr,cur[1]+dc)
            if not(0<=nb[0]<16 and 0<=nb[1]<16): continue
            if nb in prev: continue
            if nb in blocked and nb not in goals: continue
            prev[nb]=(cur,a); q.append(nb)
    return None
env=l8()
for step in range(60):
    f=np.asarray(env.frame()); av,walls,boxes,cont,movers,cours=scan(f)
    top_movers=[m for m in movers if m[0]<6]
    if not top_movers: print("TOP MOVER GONE at",step); break
    m=top_movers[0]
    blocked=set(walls)|set(boxes)|set(cont)|set(cours)|set(movers)
    # goal: cells adjacent to mover
    goals=set()
    for dr,dc in DIRS.values():
        adj=(m[0]+dr,m[1]+dc)
        if 0<=adj[0]<16 and 0<=adj[1]<16 and adj not in blocked: goals.add(adj)
    if av in goals:
        # face mover and USE
        dr,dc=m[0]-av[0],m[1]-av[1]
        face={(-1,0):1,(1,0):2,(0,-1):3,(0,1):4}[(dr,dc)]
        env.step(face); env.step(5)
        f2=np.asarray(env.frame()); _,_,_,_,mv2,_=scan(f2)
        print(step,"USE adj to",m,"movers now",mv2)
        continue
    blocked.discard(av)
    path=bfs(av,goals,blocked)
    if not path:
        print(step,"no path av",av,"m",m); env.step(1); continue
    env.step(path[0])
print("done level",env.levels_completed)
