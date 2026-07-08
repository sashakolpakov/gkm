import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
env=l8(); n0=len(env.path)
av,boxes,walls=legs._grid_scan(env)
print("av",av,"boxes",sorted(boxes),"nwalls",len(walls))
ok=legs.carry_box_to(env,(2,7),(3,12),cap=60)
print("carry ok",ok,"moves",len(env.path)-n0)
f=np.asarray(env.frame())
print("box now at (3,12)?",sorted(set(int(v) for v in np.unique(f[12:16,48:52]))))
