import l9env, perception as P
import numpy as np
def avatar(f):
    objs=[o for o in P.object_candidates(f,min_area=3) if o['color']==14]
    return objs[0]['bbox'] if objs else None
for a in [1,2,3,4]:
    env=l9env.get_l9()
    b0=avatar(np.asarray(env.frame()))
    env.step(a)
    b1=avatar(np.asarray(env.frame()))
    print(P.ACTION_NAME[a],b0,'->',b1)
