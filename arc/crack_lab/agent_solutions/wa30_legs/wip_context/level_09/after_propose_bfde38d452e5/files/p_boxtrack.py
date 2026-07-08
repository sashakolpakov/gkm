import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def boxes(f):
    out=set()
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4]
            u=set(int(v) for v in np.unique(blk))
            if 9 in u and 4 in u: out.add((R,C))
    return out
env=l8()
seen_counts={}
allpos={}
f=np.asarray(env.frame())
print("start boxes",sorted(boxes(f)))
for step in range(40):
    env.step(1 if step%2==0 else 2)
    f=np.asarray(env.frame())
    for b in boxes(f):
        allpos[b]=allpos.get(b,0)+1
print("box-cell occupancy over 40 idle steps (cell: #steps present):")
for b in sorted(allpos): print(b,allpos[b])
