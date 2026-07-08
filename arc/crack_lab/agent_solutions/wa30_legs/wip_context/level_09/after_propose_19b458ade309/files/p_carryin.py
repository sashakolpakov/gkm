import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
env=l8(); n0=len(env.path)
ok=legs.carry_box_to(env,(2,2),(3,12),cap=80)
print("carry (2,2)->(3,12) ok",ok,"moves",len(env.path)-n0,"lvl",env.levels_completed)
f=np.asarray(env.frame())
print("TOP container rows8-16 cols44-60:")
for r in f[8:16,44:60]: print(" ".join(f"{int(v):2d}" for v in r))
