import numpy as np, perception as P
from collections import deque
from solver8 import *
from l8env import l8
def m15(f): return [(int(b.bbox[0])//4,int(b.bbox[1])//4) for b in P.connected_components(f,colors=[15])]
def m12(f): return [(int(b.bbox[0])//4,int(b.bbox[1])//4) for b in P.connected_components(f,colors=[12])]
env=l8()
# go up into top arena
for a in [4,4,4,1,1,1,1]: env.step(a)
print('av',av_cell(env.frame()),'m15',m15(env.frame()))
# reactive: walk avatar onto/into nearest top 15, pressing toward it and USE, watch count
n0=len(m15(env.frame()))
for t in range(40):
    if env.terminal(): break
    f=env.frame(); a=av_cell(f); ms=[m for m in m15(f) if m[0]<6]
    if not ms: print('top15 gone t',t); break
    m=min(ms,key=lambda x:abs(x[0]-a[0])+abs(x[1]-a[1]))
    dr=m[0]-a[0]; dc=m[1]-a[1]
    if abs(dr)>=abs(dc) and dr!=0: mv=DOWN if dr>0 else UP
    elif dc!=0: mv=RIGHT if dc>0 else LEFT
    else: mv=USE
    env.step(mv)
    if abs(dr)+abs(dc)<=1:
        # adjacent: try USE and also try stepping onto it
        env.step(USE)
    if t%5==0: print('t',t,'av',av_cell(env.frame()),'m15',m15(env.frame()))
print('final m15',m15(env.frame()),'m12',m12(env.frame()),'lvl',env.levels_completed)
