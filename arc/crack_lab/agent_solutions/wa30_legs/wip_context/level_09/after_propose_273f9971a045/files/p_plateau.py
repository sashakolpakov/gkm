import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def boxcells(f):
    out=set()
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4];u=set(int(v) for v in np.unique(blk))
            if 9 in u and 4 in u: out.add((R,C))
    return out
env=l8()
for s in range(80): env.step(1 if s%2==0 else 2)
f=np.asarray(env.frame())
bc=sorted(boxcells(f))
print("boxes after 80 idle:",bc)
print("top4px",int((f[8:16,44:60]==4).sum()),"bot4px",int((f[48:60,48:60]==4).sum()))
# top-left penned still there?
print("penned present:", all(b in bc for b in [(2,1),(2,2),(3,1),(3,2)]))
