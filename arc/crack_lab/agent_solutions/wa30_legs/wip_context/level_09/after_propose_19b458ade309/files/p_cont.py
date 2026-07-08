import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
env=l8()
# top-right container interior pixels rows 9-14 cols45-58; bot-right rows49-58 cols49-58
def interior(f,r0,r1,c0,c1):
    sub=f[r0:r1,c0:c1]
    vals,cnts=np.unique(sub,return_counts=True)
    return {int(v):int(c) for v,c in zip(vals,cnts)}
for step in range(60):
    env.step(1 if step%2==0 else 2)
    if step%10==9:
        f=np.asarray(env.frame())
        print(step,"TOP",interior(f,9,15,45,59),"BOT",interior(f,49,59,49,59))
