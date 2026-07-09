import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def avc(f):
    ys,xs=np.where(f==14); return (int(ys.min())//4,int(xs.min())//4) if len(ys) else None
env=l8()
for a in [1,4,4,4,1,1,1]: env.step(a)  # to (4,4)
# go right to C12 at row4
for _ in range(8): env.step(4)
print("at",avc(np.asarray(env.frame())))
# now UP repeatedly to try entering container
for i in range(4):
    b=avc(np.asarray(env.frame())); env.step(1); a=avc(np.asarray(env.frame()))
    print("UP",b,"->",a)
    if b==a:
        f=np.asarray(env.frame()); R,C=a
        blk=f[(R-1)*4:R*4,C*4:C*4+4]
        print("  cell above:",sorted(set(int(v) for v in np.unique(blk))))
        break
