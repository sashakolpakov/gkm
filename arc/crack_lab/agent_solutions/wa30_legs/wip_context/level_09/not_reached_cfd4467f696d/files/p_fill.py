import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def boxcells(f):
    out=set()
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            u=set(int(v) for v in np.unique(blk))
            if 9 in u and 4 in u: out.add((R,C))
    return out
TOP={(r,c) for r in (2,3) for c in (11,12,13,14)}
BOT={(r,c) for r in (12,13,14) for c in (12,13,14)}
env=l8()
for step in range(120):
    env.step(1 if step%2==0 else 2)
    bc=boxcells(f:=np.asarray(env.frame()))
    t=len(bc&TOP); b=len(bc&BOT)
    if step%8==7: print(step,"top",t,"bot",b,"total_boxes",len(bc))
