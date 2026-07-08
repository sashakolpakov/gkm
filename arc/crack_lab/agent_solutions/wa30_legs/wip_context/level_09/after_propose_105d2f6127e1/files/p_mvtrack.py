import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def movers(f):
    ys,xs=np.where(f==15)
    return sorted(set((int(y)//4,int(x)//4) for y,x in zip(ys,xs)))
env=l8()
prev=None; stable=0
for s in range(120):
    env.step(1 if s%2==0 else 2)
    m=movers(np.asarray(env.frame()))
    if s%5==4: print(s,m)
