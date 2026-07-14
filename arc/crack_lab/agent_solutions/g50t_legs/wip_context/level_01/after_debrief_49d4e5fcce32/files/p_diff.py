import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
env=A.Arena('g50t')
b=env.frame().copy()
for a in [4,4,4,4]: env.step(a)
env.step(5)
f=env.frame()
ys,xs=np.where(b!=f)
print("total changed",len(ys))
for y,x in zip(ys,xs):
    print((int(y),int(x)),int(b[y,x]),"->",int(f[y,x]))
