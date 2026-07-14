import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
import numpy as np

def avatar(env):
    # interior 9 blob (col>10) that's the avatar - biggest 9 blob in maze
    bl=[b for b in P.connected_components(env.frame(),colors=[9]) if b.bbox[1]>10 and b.bbox[0]<40]
    bl.sort(key=lambda b:-b.area)
    return bl[0] if bl else None

env=A.Arena('g50t')
a=avatar(env); print("start",a.bbox,a.area)
for act in [2,2,4,4]:
    env.step(act)
    a=avatar(env); print(P.ACTION_NAME[act],a.bbox,a.area)
