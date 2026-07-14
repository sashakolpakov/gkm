import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def av(e):
    for b in P.connected_components(e.frame(),colors=[9]):
        if b.area==24: return b.bbox[:2]
    return None
def prog(env):
    c=env.clone()
    for a in [2,2,2,2]: c.step(a)  # to (32,14)
    print("at",av(c),"c8",P.color_counts(c.frame()).get(8))
    for i in range(6):
        c.step(2)
        print("down",i,"av",av(c),"c8",P.color_counts(c.frame()).get(8))
    # now what does USE do here (maybe USE eats/breaks 8 below)?
    print("--- USE at",av(c),"---")
    base=np.asarray(c.frame())
    c2=c.clone(); c2.step(5)
    d=P.frame_delta(base,c2.frame())
    print("USE delta",d['count'],d['bbox'],"av after",av(c2))
    raise SystemExit
A.run_program('g50t', prog)
