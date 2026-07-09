import l9env, perception as P
import numpy as np
env = l9env.get_l9()
def show12(f):
    # find color-12 blobs
    objs=[o for o in P.object_candidates(f,min_area=4) if o['color']==12]
    return [o['bbox'] for o in objs]
f=np.asarray(env.frame())
print("t0 c12", show12(f))
for t in range(1,10):
    env.step(5) # USE (idle-ish)
    f=np.asarray(env.frame())
    print(f"t{t} lvl{env.levels_completed} c12", show12(f))
