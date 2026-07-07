import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
env=l8()
c=env.clone()
legs._avatar_nav(c,(1,7),cap=80)
print("at",legs._grid_scan(c)[0])
c.step(legs.DOWN)  # face box (2,7)
print("after face DOWN, grabbed:",sorted(legs._grid_grabbed(c)),"av",legs._grid_scan(c)[0])
c.step(legs.USE)
print("after USE, grabbed:",sorted(legs._grid_grabbed(c)),"av",legs._grid_scan(c)[0],"boxes",sorted(legs._grid_scan(c)[1]))
