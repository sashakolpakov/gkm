import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def eights(f):
    ys,xs=np.where(f==8)
    if len(ys)==0: return None
    return (ys.min(),xs.min(),ys.max(),xs.max()),(round(ys.mean(),1),round(xs.mean(),1)),len(ys)
def av(e):
    for b in P.connected_components(e.frame(),colors=[9]):
        if 7<=b.bbox[0]<49 and b.bbox[3]!=63 and b.area>=18: return b.bbox
    return None
def prog(env):
    c=env.clone()
    print("start av",av(c),"8",eights(np.asarray(c.frame())))
    for a in [2,4,2,4,3,1]:
        c.step(a)
        print("act",P.ACTION_NAME[a],"av",av(c),"8bbox/cen/cnt",eights(np.asarray(c.frame())))
    raise SystemExit
A.run_program('g50t', prog)
