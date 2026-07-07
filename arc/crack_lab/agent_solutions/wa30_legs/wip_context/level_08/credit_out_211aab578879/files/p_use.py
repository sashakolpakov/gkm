import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
env=l8()
cD=env.clone(); cD.step(2)  # DOWN (blocked, avatar stays)
cU=env.clone(); cU.step(5)  # USE
fD=np.asarray(cD.frame()); fU=np.asarray(cU.frame())
ys,xs=np.where(fD!=fU)
print("DOWN vs USE differing px:",len(ys))
for y,x in list(zip(ys,xs))[:40]:
    print(y,x,"D",int(fD[y,x]),"U",int(fU[y,x]))
