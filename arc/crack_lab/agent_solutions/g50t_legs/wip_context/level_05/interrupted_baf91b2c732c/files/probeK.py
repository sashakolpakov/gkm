import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def av(env):
    for b in P.connected_components(env.frame(),colors=[9]):
        if b.area==24: return b.bbox
    return None
def prog(env):
    c=env.clone()
    for a in [4,4,4,4,4]:
        pre=av(c); c.step(a); print("R",pre[:2],"->",av(c)[:2] if av(c) else None,"c8",P.color_counts(c.frame()).get(8))
    raise SystemExit
A.run_program('g50t', prog)
