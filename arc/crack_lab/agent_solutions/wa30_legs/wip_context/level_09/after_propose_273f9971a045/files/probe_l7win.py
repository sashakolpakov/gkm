import numpy as np
b=np.load("l7_win_before.npy"); a=np.load("l7_win_after.npy")
ys,xs=np.where(b!=a)
from collections import Counter
print("num changed",len(ys))
print("transitions",dict(Counter((int(b[y,x]),int(a[y,x])) for y,x in zip(ys,xs))))
if len(ys): print("bbox",(int(ys.min()),int(xs.min()),int(ys.max()),int(xs.max())))
# show the changed region small
for y,x in list(zip(ys,xs))[:30]:
    print(y,x,int(b[y,x]),'->',int(a[y,x]))
