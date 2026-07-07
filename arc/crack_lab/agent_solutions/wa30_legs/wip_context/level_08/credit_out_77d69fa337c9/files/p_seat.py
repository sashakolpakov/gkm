import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
env=l8()
def dump(f,r0,r1,c0,c1):
    for r in f[r0:r1,c0:c1]: print(" ".join(f"{int(v):2d}" for v in r))
for s in range(24): env.step(1 if s%2==0 else 2)
f=np.asarray(env.frame())
print("TOP container region rows8-16 cols44-60:")
dump(f,8,16,44,60)
print("\nBOT container region rows48-60 cols44-60:")
dump(f,48,60,44,60)
