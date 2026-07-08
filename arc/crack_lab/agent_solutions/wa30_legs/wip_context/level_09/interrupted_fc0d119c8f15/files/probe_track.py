import l9env, perception as P
import numpy as np
env=l9env.get_l9()
def bl(f,col):
    return sorted([o['bbox'] for o in P.object_candidates(f,min_area=3) if o['color']==col])
for t in range(12):
    f=np.asarray(env.frame())
    print(f"t{t} c12={bl(f,12)} c15={bl(f,15)}")
    env.step(5)
