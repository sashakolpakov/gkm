import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
def avc(f):
    ys,xs=np.where(f==14)
    return (int(ys.min())//4,int(xs.min())//4) if len(ys) else None
env=l8()
# get avatar into top region first: up through gap C4-5
seq=[1,4,4,4,1,1,1]  # to (5,4) area
for a in seq: env.step(a)
print("after climb",avc(np.asarray(env.frame())))
# now try to walk right toward container across the top region row ~2
for i in range(14):
    before=avc(np.asarray(env.frame()))
    env.step(4)  # RIGHT
    after=avc(np.asarray(env.frame()))
    print("RIGHT",before,"->",after, "blocked" if before==after else "")
    if before==after: 
        # try to see what's to the right
        f=np.asarray(env.frame()); R,C=after
        blk=f[R*4:R*4+4,(C+1)*4:(C+2)*4]
        print("  right cell content:",sorted(set(int(v) for v in np.unique(blk))))
        break
