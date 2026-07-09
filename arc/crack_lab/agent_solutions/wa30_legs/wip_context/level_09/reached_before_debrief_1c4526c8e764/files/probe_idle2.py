import l9env, perception as P
import numpy as np
env=l9env.get_l9()
base=np.asarray(env.frame()).copy()
for t in range(30):
    env.step(5)
f=np.asarray(env.frame())
d=P.frame_delta(base,f)
print("after 30 USE, changed cells:",d['count'],"lvl",env.levels_completed)
# show changes excluding avatar region and mover trail? just list colors involved
ys,xs=np.where(base!=f)
from collections import Counter
c=Counter((int(base[y,x]),int(f[y,x])) for y,x in zip(ys,xs))
print("transitions (from->to):",dict(c))
