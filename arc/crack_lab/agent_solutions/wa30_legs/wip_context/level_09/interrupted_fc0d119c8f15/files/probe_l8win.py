import numpy as np
b=np.load("l8_win_before.npy"); a=np.load("l8_win_after.npy")
print("before colors",{int(v):int(c) for v,c in zip(*np.unique(b,return_counts=True))})
print("after colors",{int(v):int(c) for v,c in zip(*np.unique(a,return_counts=True))})
ys,xs=np.where(b!=a)
from collections import Counter
print("transitions",dict(Counter((int(b[y,x]),int(a[y,x])) for y,x in zip(ys,xs))))
print("num changed",len(ys),"bbox",(ys.min(),xs.min(),ys.max(),xs.max()) if len(ys) else None)
