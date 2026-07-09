import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
path=json.load(open('/tmp/l8_minagent.json'))
def avc(f):
    ys,xs=np.where(f==14); return (int(ys.min())//4,int(xs.min())//4)
for park in ["bottomleft","hold"]:
    env=l8()
    for a in path[:20]: env.step(a)
    f=np.asarray(env.frame()); print(park,"avatar after prefix",avc(f))
    base=env.levels_completed
    # park: drive to bottom-left corner
    for a in [3,3,3,3,3,3,2,2,2,2,2]: 
        if not env.terminal(): env.step(a)
    def fill(f): return int((f[8:16,44:60]==4).sum()),int((f[48:60,48:60]==4).sum())
    won=False
    for i in range(105):
        if env.terminal(): print("term",i);break
        env.step(3 if i%2==0 else 4)  # wiggle in corner
        if env.levels_completed>base: print("WIN total",len(env.path));won=True;break
        if i%15==14: f=np.asarray(env.frame()); print(" ",len(env.path),"fill",fill(f),"av",avc(f))
    print(park,"final",env.levels_completed)
    break
