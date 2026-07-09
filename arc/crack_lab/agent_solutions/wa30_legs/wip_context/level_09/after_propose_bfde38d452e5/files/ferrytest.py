import numpy as np
from collections import deque
from solver8 import *  # helpers
from l8env import l8

def rigid_carry(env, goal_av, cap=40):
    # carry currently-grabbed box; move avatar to goal_av keeping both cells clear
    for _ in range(cap):
        if env.terminal(): return False
        f=env.frame(); av=av_cell(f); g=grabbed_cell(f)
        if g is None: return False
        if av==goal_av: return True
        off=(g[0]-av[0],g[1]-av[1])
        # BFS avatar to goal_av; a move ok if new av cell and new box cell both empty (box may be current)
        def occ(ff,R,C):
            return cell_empty(ff,R,C)
        # custom: both av-target and box-target empty
        q=deque([av]); prev={av:None}
        found=None
        while q:
            cur=q.popleft()
            if cur==goal_av: found=cur;break
            for a,(dr,dc) in DIRS.items():
                nb=(cur[0]+dr,cur[1]+dc); bx=(nb[0]+off[0],nb[1]+off[1])
                if nb in prev: continue
                if not(0<=nb[0]<16 and 0<=nb[1]<16 and 0<=bx[0]<16 and 0<=bx[1]<16): continue
                okav = cell_empty(f,nb[0],nb[1]) or nb==g
                okbx = cell_empty(f,bx[0],bx[1]) or bx==av
                if okav and okbx:
                    prev[nb]=(cur,a); q.append(nb)
        if found is None: return False
        # first action
        path=[];n=found
        while prev[n]: p,a=prev[n];path.append(a);n=p
        path=path[::-1]
        before=av
        env.step(path[0])
        if av_cell(env.frame())==before: return False
    return av_cell(env.frame())==goal_av

def grab_box(env, box, cap=40):
    # nav to a cell adjacent to box and face+grab
    f=env.frame(); av=av_cell(f)
    adj={}
    for a,(dr,dc) in DIRS.items():
        ac=(box[0]-dr,box[1]-dc)
        if cell_empty(f,ac[0],ac[1]): adj[ac]=a
    if not adj: return False
    path,reached=bfs(f,av,set(adj.keys()))
    if path is None and av not in adj: return False
    if path:
        for a in path:
            if env.terminal(): return False
            env.step(a)
    f2=env.frame(); av2=av_cell(f2)
    if av2 in adj:
        env.step(adj[av2]); env.step(USE)
        return grabbed_cell(env.frame()) is not None
    return False

env=l8()
# fixed top-left boxes = box cells with col<=2 and row<=3
def fixed(f): return sorted([b for b in box_cells(f) if b[1]<=2 and b[0]<=3])
drops=[(2,7),(2,8),(3,7),(3,8)]  # courier lane in top-mid arena
for i in range(4):
    f=env.frame(); fx=fixed(f)
    if not fx: print('no more fixed');break
    box=fx[0]
    ok=grab_box(env,box)
    if not ok: print('grab fail',box,'steps',len(env.path)-466);continue
    # carry to a drop: bring box to drops[i]; goal_av so that box lands on drop
    f=env.frame(); av=av_cell(f); g=grabbed_cell(f); off=(g[0]-av[0],g[1]-av[1])
    drop=drops[i]; goal_av=(drop[0]-off[0],drop[1]-off[1])
    okc=rigid_carry(env,goal_av)
    env.step(USE)  # release
    print('ferried',box,'->',drop,'carry',okc,'steps',len(env.path)-466,'nbox',len(box_cells(env.frame())))
# idle remaining budget
while not env.terminal() and env.levels_completed==7 and len(env.path)-466<140:
    env.step(USE)
print('FINAL lvl',env.levels_completed,'steps',len(env.path)-466,'fixed_left',len(fixed(env.frame())),'nbox',len(box_cells(env.frame())))
