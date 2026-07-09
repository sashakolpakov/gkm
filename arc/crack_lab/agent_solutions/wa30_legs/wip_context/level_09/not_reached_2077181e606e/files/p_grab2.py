import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
env=l8()
c=env.clone()
legs._avatar_nav(c,(1,2),cap=80)
print("at",legs._grid_scan(c)[0])
b=legs._grid_scan(c)[1]; print("(2,2) is box?",(2,2) in b)
c.step(legs.DOWN)
print("after DOWN: av",legs._grid_scan(c)[0],"grabbed",sorted(legs._grid_grabbed(c)))
c.step(legs.USE)
print("after USE: av",legs._grid_scan(c)[0],"grabbed",sorted(legs._grid_grabbed(c)))
# try to carry up
c.step(legs.UP)
print("after UP: av",legs._grid_scan(c)[0],"grabbed",sorted(legs._grid_grabbed(c)))
