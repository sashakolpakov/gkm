import l9env, perception as P
import numpy as np
env=l9env.get_l9()
def c15(f):
    b=[o['bbox'] for o in P.object_candidates(f,min_area=3) if o['color']==15]
    return b[0] if b else None
prev=None
for t in range(66):
    f=np.asarray(env.frame())
    cur=c15(f)
    cell=(cur[0]//4,cur[1]//4) if cur else None
    print(f"t{t} c15cell={cell}")
    env.step(5)
