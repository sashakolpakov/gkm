import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
env=l8()
c=env.clone()
ok=legs._avatar_nav(c,(1,7),cap=80)
print("nav to (1,7):",ok,"avatar now",legs._grid_scan(c)[0])
# check obstacles reader
av,blocked=legs._obstacles_grid(env)
print("av",av)
print("(6,4) blocked?",(6,4) in blocked,"(6,5)?",(6,5) in blocked)
print("(1,7) blocked?",(1,7) in blocked)
# is there a static bfs path?
from legs import _bfs_path
p=_bfs_path(av,(1,7),blocked)
print("bfs path len",None if p is None else len(p))
