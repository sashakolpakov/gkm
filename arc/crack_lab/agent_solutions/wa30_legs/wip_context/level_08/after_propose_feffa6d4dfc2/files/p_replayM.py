import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def cells(f,c):
    ys,xs=np.where(f==c); return sorted(set((int(y)//4,int(x)//4) for y,x in zip(ys,xs)))
path=json.load(open('/tmp/l8_minagent.json'))
env=l8()
prev15=int((np.asarray(env.frame())==15).sum())
name={1:'U',2:'D',3:'L',4:'R',5:'USE'}
for i,a in enumerate(path):
    env.step(a); f=np.asarray(env.frame())
    n15=int((f==15).sum())
    av=cells(f,14)
    if n15!=prev15:
        print(f"step{i} act{name[a]} av{av} 15px:{prev15}->{n15} movers{cells(f,15)}")
        prev15=n15
print("final 15px",prev15,"path",[name[a] for a in path])
