import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
path=json.load(open('/tmp/l8_minagent.json'))
env=l8()
for a in path[:20]: env.step(a)
for i in range(100): env.step(1 if i%2==0 else 2)
f=np.asarray(env.frame())
def boxcells(f):
    out=set()
    for R in range(16):
        for C in range(16):
            blk=f[R*4:R*4+4,C*4:C*4+4];u=set(int(v) for v in np.unique(blk))
            if 9 in u and 4 in u: out.add((R,C))
    return sorted(out)
print("boxes",boxcells(f))
print("couriers",[(int(y)//4,int(x)//4) for y,x in zip(*np.where(f==12))][:8])
print("TOP:")
for r in f[8:16,44:60]: print(" ".join(f"{int(v):2d}" for v in r))
print("BOT:")
for r in f[48:60,44:60]: print(" ".join(f"{int(v):2d}" for v in r))
