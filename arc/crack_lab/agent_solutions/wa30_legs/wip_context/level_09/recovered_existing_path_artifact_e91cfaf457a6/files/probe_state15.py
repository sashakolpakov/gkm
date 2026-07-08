import l9env, legs, perception as P
import numpy as np
env=l9env.get_l9()
legs.chase_and_clear(env,15,lambda m:True,cap=50)
f=np.asarray(env.frame())
print("steps",len(env.path)-588,"n7",int((f==7).sum()))
print("    "+''.join(str(c%10) for c in range(64)))
for r in range(40):
    print(f'{r:2d}: '+''.join(f'{int(v):x}' if v else '.' for v in f[r]))
# 9-frames interiors
for b in P.connected_components(f,colors={9}):
    r0,c0,r1,c1=b.bbox
    if (r1-r0)>=3 and (c1-c0)>=3:
        interior=f[r0+1:r1,c0+1:c1]
        print("9frame",b.bbox,{int(v):int((interior==v).sum()) for v in np.unique(interior)})
