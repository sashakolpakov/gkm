import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def av(env):
    for b in P.connected_components(env.frame(),colors=[9]):
        if b.area==24: return b.bbox[:2]
    return None
def prog(env):
    c=env.clone()
    for a in [2,2,2,2]:  # down x4 -> (32,14)?
        c.step(a); print("pos",av(c),"c8",P.color_counts(c.frame()).get(8))
    print("try DOWN from",av(c))
    for a in [2,2,2]:
        pre=av(c); c.step(a); print(pre,"->",av(c))
    raise SystemExit
A.run_program('g50t', prog)
