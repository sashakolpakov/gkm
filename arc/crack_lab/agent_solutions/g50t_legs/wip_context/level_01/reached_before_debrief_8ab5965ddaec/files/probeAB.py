import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def av(e):
    for b in P.connected_components(e.frame(),colors=[9]):
        if 7<=b.bbox[0]<49 and b.bbox[3]!=63 and b.area>=18: return b.bbox
    return None
def c8(e): return P.color_counts(e.frame()).get(8)
def prog(env):
    c=env.clone()
    for a in [4,4,4,4]: c.step(a)
    print("at (8,38) av",av(c),"c8",c8(c))
    for a in (1,2,3,4,5):
        cc=c.clone(); cc.step(a)
        print(" move",P.ACTION_NAME[a],"->av",av(cc),"c8",c8(cc))
    raise SystemExit
A.run_program('g50t', prog)
