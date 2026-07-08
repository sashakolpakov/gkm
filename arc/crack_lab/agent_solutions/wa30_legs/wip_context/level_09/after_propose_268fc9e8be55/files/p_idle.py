import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def cells(f,color):
    ys,xs=np.where(f==color)
    if not len(ys): return []
    return sorted(set((int(y)//4,int(x)//4) for y,x in zip(ys,xs)))
def boxes(f):
    out=set()
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            u=set(int(v) for v in np.unique(blk))
            if 9 in u and 4 in u: out.add((R,C))
    return sorted(out)
env=l8()
# idle with USE? no, USE grabs. Use a wiggle in place: LEFT/RIGHT? avatar moves. 
# Let's idle by pressing UP then DOWN alternately far from boxes.
for step in range(20):
    a = 1 if step%2==0 else 2
    env.step(a)
    f=np.asarray(env.frame())
    print(step,"av",cells(f,14),"mv15",cells(f,15),"c12",cells(f,12),"boxes",len(boxes(f)))
