import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
def boxcells(f):
    out=set()
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]; u=set(int(v) for v in np.unique(blk))
            if 9 in u and 4 in u: out.add((R,C))
    return out
env=l8(); n0=len(env.path)
# try filling top container cells
targets=[(2,11),(2,12),(2,13),(2,14),(3,11),(3,12),(3,13),(3,14)]
legs.fill_targets_nearest_first(env, targets)
print("after top fill: moves",len(env.path)-n0,"level",env.levels_completed)
f=np.asarray(env.frame()); print("boxes in top region:",sorted(b for b in boxcells(f) if b[0]<6))
