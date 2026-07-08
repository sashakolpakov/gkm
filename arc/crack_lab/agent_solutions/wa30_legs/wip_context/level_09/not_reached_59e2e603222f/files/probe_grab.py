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
env=l8()
f=env.frame()
# go to left-adjacent of box (2,1) rows8-11 cols4-7 => avatar at (8,0)
path=bfs(f,avpos(f),(8,0)); print('nav',len(path))
for a in path: env.step(a)
f=env.frame(); print('avatar',avpos(f))
print('region rows7-12 cols0-12 BEFORE grab:')
for r in range(7,13): print(''.join('%2d'%f[r,c] for c in range(0,13)))
env.step(4); env.step(5)  # face right, grab
f=env.frame(); print('AFTER grab avatar',avpos(f))
for r in range(7,13): print(''.join('%2d'%f[r,c] for c in range(0,13)))
# carry UP a couple
for i in range(2):
    env.step(1)
f=env.frame(); print('AFTER carry up avatar',avpos(f))
for r in range(4,13): print(''.join('%2d'%f[r,c] for c in range(0,13)))
env.step(5); f=env.frame(); print('AFTER release')
for r in range(4,13): print(''.join('%2d'%f[r,c] for c in range(0,13)))
print('lvl',env.levels_completed)
