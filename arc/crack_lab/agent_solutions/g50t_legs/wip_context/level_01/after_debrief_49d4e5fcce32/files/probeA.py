import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def av(env):
    for b in P.connected_components(env.frame(),colors=[9]):
        if b.area==24: return b.bbox
    return None
def leg(f):
    return f[0:6,0:9].tolist()

def prog(env):
    c=env.clone(); c.step(2) # to (14,14)
    print("at",av(c))
    base=np.asarray(c.frame())
    for a in (1,2,3,4,5):
        cc=c.clone(); cc.step(a)
        f=np.asarray(cc.frame())
        d=P.frame_delta(base,f)
        print("act",a,"count",d['count'],"avatar",av(cc),"lvl",cc.levels_completed,"legendchanged",leg(base)!=leg(f))
    raise SystemExit
A.run_program('g50t', prog)
