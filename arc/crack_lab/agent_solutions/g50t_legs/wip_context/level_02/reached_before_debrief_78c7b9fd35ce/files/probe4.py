import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def avatar(env):
    for b in P.connected_components(env.frame(),colors=[9]):
        if b.area==24 or (18<=b.area<=26 and b.bbox[0]<40):
            return b
    return None

def prog(env):
    c=env.clone(); c.step(2)  # down
    for b in P.connected_components(c.frame(),colors=[9]):
        print("after DOWN",b.bbox,b.area)
    c=env.clone(); c.step(4)
    print("---right---")
    for b in P.connected_components(c.frame(),colors=[9]):
        print(b.bbox,b.area)
    raise SystemExit
A.run_program('g50t', prog)
