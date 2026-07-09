import l9env, perception as P
import numpy as np
env=l9env.get_l9()
f=np.asarray(env.frame())
# find 9-connected components as frames, report interior composition
blobs=P.connected_components(f,colors={9})
for b in blobs:
    r0,c0,r1,c1=b.bbox
    if (r1-r0)>=3 and (c1-c0)>=3:
        interior=f[r0+1:r1,c0+1:c1]
        vals={int(v):int((interior==v).sum()) for v in np.unique(interior)}
        print("9-frame bbox",b.bbox,"size",(r1-r0+1,c1-c0+1),"interior",vals)
