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
            elif 9 in u or (2 in u and 4 not in u): cont.add((R,C))
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
for step in range(30):
    f=np.asarray(env.frame()); av,walls,boxes,cont,movers,cours=scan(f)
    m=[mm for mm in movers if mm[0]<6][0]
    blocked=set(walls)|set(boxes)|set(cont)|set(cours)|set(movers); blocked.discard(av)
    goals=set()
    for dr,dc in DIRS.values():
        adj=(m[0]+dr,m[1]+dc)
        if 0<=adj[0]<16 and 0<=adj[1]<16 and adj not in blocked: goals.add(adj)
    path=bfs(av,goals,blocked)
    print(step,"av",av,"m",m,"pathlen",None if path is None else len(path))
    if path is None: env.step(1); continue
    if not path:
        dr,dc=m[0]-av[0],m[1]-av[1]
        env.step({(-1,0):1,(1,0):2,(0,-1):3,(0,1):4}[(dr,dc)]); env.step(5)
        continue
    env.step(path[0])
