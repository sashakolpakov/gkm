# Test: extract penned boxes then yield -> win?
import sys; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
env=l8(); base=env.levels_completed; n0=len(env.path)
# Use carry_box_to but it treats containers as free (bug). Let's just try pulling
# penned boxes to open top cells via carry_box_to with a proper drop in open area.
# penned: (2,1),(2,2),(3,1),(3,2). Open drop cells in top region: (1,5),(1,6),(0,5)...
# But carry_box_to uses _grid_scan (walls5 only). Containers not walls -> ok for top-left.
for box,drop in [((2,2),(1,6)),((2,1),(1,5)),((3,2),(1,7)),((3,1),(0,6))]:
    ok=legs.carry_box_to(env,box,drop,cap=40)
    print("carry",box,"->",drop,"ok",ok,"moves",len(env.path)-n0,"lvl",env.levels_completed)
    if env.levels_completed>base: break
# yield
for _ in range(30):
    if env.terminal() or env.levels_completed>base: break
    env.step(1 if _%2==0 else 2)
print("final lvl",env.levels_completed,"moves",len(env.path)-n0)
