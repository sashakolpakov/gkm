import numpy as np, perception as P
from solver8 import *
from l8env import l8
def m15(f): return [(int(b.bbox[0])//4,int(b.bbox[1])//4) for b in P.connected_components(f,colors=[15])]
env=l8()
# grab the fixed box (2,1) (mover-proof, easy) then carry toward a 15
for a in [4,4,4,1,1,1,1]: env.step(a)
# nav to left of a top box in the open arena; simpler: grab box at (2,7)-ish if present
f=env.frame()
# just grab nearest box via solver8 grab logic
from collections import deque
def grab_nearest(env):
    f=env.frame(); av=av_cell(f); bs=sorted(box_cells(f),key=lambda b:abs(b[0]-av[0])+abs(b[1]-av[1]))
    for box in bs[:5]:
        adj={}
        for a,(dr,dc) in DIRS.items():
            ac=(box[0]-dr,box[1]-dc)
            if cell_empty(f,ac[0],ac[1]): adj[ac]=a
        if not adj: continue
        path,_=bfs(f,av,set(adj.keys()))
        if path is None and av not in adj: continue
        if path:
            for a in path:
                if env.terminal(): return False
                env.step(a)
        av2=av_cell(env.frame())
        if av2 in adj:
            env.step(adj[av2]); env.step(USE)
            return grabbed_cell(env.frame()) is not None
    return False
ok=grab_nearest(env)
print('grabbed?',ok,'g',grabbed_cell(env.frame()),'m15',m15(env.frame()))
n0=len(m15(env.frame()))
# now chase a 15 with the carried box: move so box cell approaches a 15, push into it
for t in range(30):
    if env.terminal(): break
    f=env.frame(); av=av_cell(f); g=grabbed_cell(f)
    if g is None: print('lost box t',t); break
    ms=[m for m in m15(f) if m[0]<8]
    if not ms: print('a top 15 GONE t',t,'->',len(m15(f))); break
    m=min(ms,key=lambda x:abs(x[0]-g[0])+abs(x[1]-g[1]))
    dr=m[0]-g[0]; dc=m[1]-g[1]
    off=(g[0]-av[0],g[1]-av[1])
    if abs(dr)>=abs(dc) and dr!=0: a=DOWN if dr>0 else UP
    elif dc!=0: a=RIGHT if dc>0 else LEFT
    else: a=USE
    env.step(a)
print('final m15 count',len(m15(env.frame())),'was',n0,'lvl',env.levels_completed)
