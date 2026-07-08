import numpy as np
from collections import deque
from l8env import l8
SOLID={2,4,5,7,9,12,15}
def avpos(f):
    ys,xs=np.where((f==14)|(f==0))
    return (int(ys.min()),int(xs.min()))  # top-left of body
def freeat(f,r,c):
    # a 4x4 block starting (r,c) all passable (1 or 0)?
    if r<0 or c<0 or r+4>64 or c+4>64: return False
    blk=f[r:r+4,c:c+4]
    return bool(np.all((blk==1)|(blk==0)))
def bfs(f,start,goal):
    # move by 4px steps; body top-left positions on 4-grid
    q=deque([start]); prev={start:None}
    while q:
        cur=q.popleft()
        if cur==goal:
            path=[]; n=cur
            while prev[n]: p,a=prev[n]; path.append(a); n=p
            return path[::-1]
        r,c=cur
        for a,(dr,dc) in {1:(-4,0),2:(4,0),3:(0,-4),4:(0,4)}.items():
            nr,nc=r+dr,c+dc; nb=(nr,nc)
            if nb in prev: continue
            if not freeat(f,nr,nc): continue
            prev[nb]=(cur,a); q.append(nb)
    return None
env=l8()
f=env.frame()
print('avatar',avpos(f))
# box at rows8-11 cols36-39 (cell 2,9). left-adjacent avatar top-left = (8,32)
goal=(8,32)
path=bfs(f,avpos(f),goal)
print('path len',len(path) if path else None)
for a in path: env.step(a)
print('now avatar',avpos(env.frame()))
def boxcnt(f):
    # count 9-cells that are box-cores (have a 4 or 0 in 4x4 around) -- approx total 9
    return int((f==9).sum())
def socket9(f): return int((f[9:15,45:59]==9).sum())
print('before grab: total9',boxcnt(env.frame()),'socket9',socket9(env.frame()))
env.step(4)  # face right toward box
env.step(5)  # grab
f=env.frame()
# check box border near cols36-39
print('after grab region rows8-11 cols34-40:')
for r in range(8,12): print(''.join('%2d'%f[r,c] for c in range(32,42)))
print('avatar',avpos(f))
# carry right into socket
for i in range(4):
    env.step(4)
    print('carry',i,'av',avpos(env.frame()),'socket9',socket9(env.frame()),'total9',boxcnt(env.frame()))
env.step(5) # release
print('after release socket9',socket9(env.frame()),'total9',boxcnt(env.frame()),'lvl',env.levels_completed)
