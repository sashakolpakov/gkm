import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
from collections import deque
DIRS={1:(-1,0),2:(1,0),3:(0,-1),4:(0,1)}
FACE={(-1,0):1,(1,0):2,(0,-1):3,(0,1):4}
def scan(f):
    walls=set();boxes=set();movers=set();cours=set();av=None
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4];u=set(int(v) for v in np.unique(blk))
            if 14 in u: av=(R,C)
            if 15 in u: movers.add((R,C))
            if 12 in u: cours.add((R,C))
            if int((blk==5).sum())>=8: walls.add((R,C))
            if 9 in u and 4 in u and 0 not in u: boxes.add((R,C))
    return av,walls,boxes,movers,cours
def bfs(start,goals,blocked):
    if start in goals: return []
    q=deque([start]);prev={start:None}
    while q:
        cur=q.popleft()
        for a,(dr,dc) in DIRS.items():
            nb=(cur[0]+dr,cur[1]+dc)
            if not(0<=nb[0]<16 and 0<=nb[1]<16) or nb in prev: continue
            if nb in blocked and nb not in goals: continue
            prev[nb]=(cur,a)
            if nb in goals:
                p=[];n=nb
                while prev[n] is not None: pp,a2=prev[n];p.append(a2);n=pp
                return p[::-1]
            q.append(nb)
    return None
def clear_movers(env, cap=60):
    for _ in range(cap):
        f=np.asarray(env.frame()); av,walls,boxes,movers,cours=scan(f)
        if not movers: return True
        # adjacent mover? USE
        done=False
        for m in movers:
            d=(m[0]-av[0],m[1]-av[1])
            if d in FACE:
                env.step(FACE[d]); env.step(5); done=True; break
        if done: continue
        # move toward nearest mover, but step onto a cell adjacent to it (to be ready)
        m=min(movers,key=lambda m:abs(m[0]-av[0])+abs(m[1]-av[1]))
        goals=set()
        for dr,dc in DIRS.values():
            adj=(m[0]+dr,m[1]+dc)
            if 0<=adj[0]<16 and 0<=adj[1]<16 and adj not in walls and adj not in boxes and adj not in cours: goals.add(adj)
        blocked=set(walls)|set(boxes)|set(cours)|(set(movers)-{m}); blocked.discard(av)
        path=bfs(av,goals,blocked)
        if not path: env.step(1); continue
        env.step(path[0])
    return not scan(np.asarray(env.frame()))[3]
env=l8()
ok=clear_movers(env)
print("cleared",ok,"moves",len(env.path)-466,"movers left",scan(np.asarray(env.frame()))[3])
