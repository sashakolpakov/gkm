import l9env, legs, perception as P
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
def c12count(f): return int((f==12).sum())
# approach stationary c12 at cell (1,11): stand at (2,11) facing UP then USE
env=l9env.get_l9()
av,head,boxes,cour,walls=legs._cells(env)
blocked=set(walls)|set(boxes)
# target adjacent cell (2,11)
path=bfs(av,(2,11),blocked)
print("path to (2,11)",path)
for a in path: 
    if env.terminal():break
    env.step(a)
f=np.asarray(env.frame())
print("at",legs._cells(env)[0],"c12=",c12count(f),"term",env.terminal())
# face up toward (1,11) and USE
env.step(1)  # UP will try to move into (1,11); if blocked, stays but faces up
f=np.asarray(env.frame()); print("after UP av",legs._cells(env)[0],"c12=",c12count(f))
env.step(5)
f=np.asarray(env.frame()); print("after USE c12=",c12count(f),"term",env.terminal(),"lvl",env.levels_completed)
# region
for r in range(0,12):
    print(f'{r:2d}: '+''.join(f'{int(f[r,c]):x}' if f[r,c] else '.' for c in range(40,52)))
