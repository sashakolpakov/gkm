import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
env=l8()
def c3(f): return int((f==3).sum())
legs.carry_box_to(env,(2,2),(3,11),cap=80)
f=np.asarray(env.frame()); print("after seat1 c3px",c3(f),"lvl",env.levels_completed)
# idle
for i in range(12): env.step(2)
f=np.asarray(env.frame()); print("after idle c3px",c3(f))
print("TOP:")
for r in f[8:16,44:60]: print(" ".join(f"{int(v):2d}" for v in r))
