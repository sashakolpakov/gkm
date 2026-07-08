import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
path=json.load(open('/tmp/l8_minagent.json'))
env=l8()
for a in path[:20]: env.step(a)
# nav avatar to middle band far-left, out of both couriers' regions
legs._walk_avatar_to(env,(8,0),legs._obstacles_grid,40)
def avc(f):
    ys,xs=np.where(f==14); return (int(ys.min())//4,int(xs.min())//4)
f=np.asarray(env.frame()); print("parked at",avc(f),"moves",len(env.path)-466)
base=env.levels_completed
def fill(f): return int((f[8:16,44:60]==4).sum()),int((f[48:60,48:60]==4).sum())
mx=(0,0)
for i in range(200):
    if env.terminal(): print("crash at",len(env.path)-466);break
    # wiggle in middle band without leaving it: LEFT (into wall) / RIGHT
    env.step(3 if i%2==0 else 4)
    if env.levels_completed>base: print("WIN",len(env.path)-466);break
    t,b=fill(np.asarray(env.frame())); mx=(max(mx[0],t),max(mx[1],b))
print("maxfill",mx,"final lvl",env.levels_completed)
