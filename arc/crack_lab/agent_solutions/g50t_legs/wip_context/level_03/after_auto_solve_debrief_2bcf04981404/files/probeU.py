import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    c=env.clone()
    for a in [4,4,4]: c.step(a)  # to (8,32)
    base=np.asarray(c.frame()).copy()
    c.step(4)  # eat move -> (8,38)
    f=np.asarray(c.frame())
    ys,xs=np.where(base!=f)
    print("total changed",len(ys))
    for y,x in zip(ys,xs):
        print(y,x,base[y,x],"->",f[y,x])
    raise SystemExit
A.run_program('g50t', prog)
