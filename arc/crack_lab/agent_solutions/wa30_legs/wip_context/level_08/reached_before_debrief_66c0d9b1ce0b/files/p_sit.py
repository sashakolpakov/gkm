import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
def movers(f):
    ys,xs=np.where(f==15); return set((int(y)//4,int(x)//4) for y,x in zip(ys,xs))
def avc(f):
    ys,xs=np.where(f==14);return (int(ys.min())//4,int(xs.min())//4)
def sit_clear(env, spot, face, cap=40):
    legs._walk_avatar_to(env, spot, legs._obstacles_grid, 30)
    for _ in range(cap):
        f=np.asarray(env.frame())
        if avc(f)!=spot:
            legs._walk_avatar_to(env, spot, legs._obstacles_grid, 10); continue
        env.step(face)  # face toward gap approach
        env.step(5)     # USE
        # check if a mover got cleared (count drop)
    return
env=l8()
# top gap chokepoint: sit at (6,4) facing UP; bottom gap: (10,9) facing... 
f=np.asarray(env.frame()); print("start movers",movers(f))
# sit at (6,4) face up, spam use, watch mover count
legs._walk_avatar_to(env,(6,4),legs._obstacles_grid,30)
print("at",avc(np.asarray(env.frame())))
for i in range(30):
    m0=len(movers(np.asarray(env.frame())))
    env.step(1); env.step(5)  # face up, use
    f=np.asarray(env.frame()); m1=movers(f)
    if len(m1)*16 < m0*16:  # dummy
        pass
    print(i,"movers",m1,"av",avc(f))
    if len(m1)<2: print("removed one, moves",len(env.path)-466); 
    if not any(mm[0]<6 for mm in m1): print("TOP MOVER GONE at",i);break
