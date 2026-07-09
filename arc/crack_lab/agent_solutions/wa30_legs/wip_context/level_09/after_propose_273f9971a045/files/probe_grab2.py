import numpy as np
from collections import deque
from l8env import l8
def avpos(f):
    ys,xs=np.where((f==14)|(f==0)); return (int(ys.min()),int(xs.min()))
def freeat(f,r,c):
    if r<0 or c<0 or r+4>64 or c+4>64: return False
    blk=f[r:r+4,c:c+4]; return bool(np.all((blk==1)|(blk==0)|(blk==14)))
def bfs(f,start,goal):
    q=deque([start]); prev={start:None}
    while q:
        cur=q.popleft()
        if cur==goal:
            path=[];n=cur
            while prev[n]: p,a=prev[n];path.append(a);n=p
            return path[::-1]
        r,c=cur
        for a,(dr,dc) in {1:(-4,0),2:(4,0),3:(0,-4),4:(0,4)}.items():
            nb=(r+dr,c+dc)
            if nb in prev or not freeat(f,r+dr,c+dc): continue
            prev[nb]=(cur,a); q.append(nb)
    return None
def show(f,tag):
    print('==',tag,'cnt4',int((f==4).sum()),'cnt9',int((f==9).sum()),'cnt2',int((f==2).sum()))
    for r in range(0,16): print(''.join('%2d'%f[r,c] for c in range(0,16)))
env=l8()
f=env.frame()
path=bfs(f,avpos(f),(8,0))
for a in path: env.step(a)
show(env.frame(),'arrived left of box(2,1)')
env.step(4)  # face right
show(env.frame(),'faced right')
env.step(5)  # grab
show(env.frame(),'grabbed')
env.step(3)  # carry LEFT (into open cols0-3)
show(env.frame(),'carried left')
env.step(5)  # release
show(env.frame(),'released')
print('lvl',env.levels_completed)
