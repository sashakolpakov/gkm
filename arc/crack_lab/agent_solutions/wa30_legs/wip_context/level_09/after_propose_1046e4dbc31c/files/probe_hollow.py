import l9env, perception as P
import numpy as np
env=l9env.get_l9()
f=np.asarray(env.frame())
for col in [9,4,2,5,12]:
    for b in P.connected_components(f,colors={col}):
        r0,c0,r1,c1=b.bbox
        h,w=r1-r0+1,c1-c0+1
        if h>=3 and w>=3:
            interior=f[r0+1:r1,c0+1:c1]
            vals={int(v):int((interior==v).sum()) for v in np.unique(interior)}
            # only report if interior has background (1) -> possible empty socket
            if 1 in vals:
                print(f"col{col} frame bbox{b.bbox} size{(h,w)} interior{vals}")
