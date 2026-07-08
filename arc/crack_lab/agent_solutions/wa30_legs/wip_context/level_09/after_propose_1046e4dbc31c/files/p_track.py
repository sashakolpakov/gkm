import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def cells(f,color):
    ys,xs=np.where(f==color)
    if not len(ys): return []
    # cluster into cells
    return sorted(set((int(y)//4,int(x)//4) for y,x in zip(ys,xs)))
env=l8()
for a in [1,2,3,4,5]:
    c=env.clone(); c.step(a)
    f=np.asarray(c.frame())
    print("act",a,"avatar14",cells(f,14),"head0",cells(f,0),"mover15",cells(f,15),"cour12",cells(f,12))
