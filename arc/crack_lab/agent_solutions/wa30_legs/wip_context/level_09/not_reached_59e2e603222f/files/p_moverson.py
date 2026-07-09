import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
env=l8(); base=env.levels_completed
TL=(slice(8,20),slice(4,16)); BL=(slice(48,60),slice(12,24))
TC=(slice(8,16),slice(44,60)); BC=(slice(48,60),slice(48,60))
def fills(f): return (int((f[TL]==4).sum()),int((f[BL]==4).sum()),int((f[TC]==4).sum()),int((f[BC]==4).sum()))
# free penned boxes into open
for b,d in [((2,2),(1,6)),((2,1),(1,5)),((3,2),(0,6)),((3,1),(0,5))]:
    legs.carry_box_to(env,b,d,cap=40)
    if env.levels_completed>base: print("WIN during free");break
print("after free moves",len(env.path)-466,"lvl",env.levels_completed)
# park middle
legs._walk_avatar_to(env,(8,0),legs._obstacles_grid,30)
mx=[0,0,0,0]
for i in range(120):
    if env.terminal(): print("crash",len(env.path)-466);break
    env.step(3 if i%2==0 else 4)
    if env.levels_completed>base: print("WIN yield",len(env.path)-466);break
    fs=fills(np.asarray(env.frame())); mx=[max(a,b) for a,b in zip(mx,fs)]
print("maxfills TL,BL,TC,BC",mx,"final",env.levels_completed)
