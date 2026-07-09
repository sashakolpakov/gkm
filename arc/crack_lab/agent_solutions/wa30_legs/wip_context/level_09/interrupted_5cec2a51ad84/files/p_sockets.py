import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
env=l8(); f=np.asarray(env.frame())
# look for single-cell sockets: border one color, core another
for R in range(16):
    for C in range(16):
        blk=f[R*4:R*4+4,C*4:C*4+4]
        border=set(int(v) for v in np.concatenate([blk[0],blk[-1],blk[:,0],blk[:,-1]]))
        core=set(int(v) for v in blk[1:3,1:3].flatten())
        if len(border)==1 and len(core)==1 and border!=core:
            print(f"R{R}C{C} border{border} core{core}")
