import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    c=env.clone()
    for a in [4,4,4,4]: c.step(a)  # (8,38) eaten
    base=np.asarray(c.frame()).copy()
    c2=c.clone(); c2.step(5)
    ys,xs=np.where(base!=np.asarray(c2.frame()))
    print("USE@(8,38) changed",len(ys),"lvl",c2.levels_completed)
    f=np.asarray(c2.frame())
    for y,x in zip(ys,xs):
        if not (y<7 or y==63):  # skip legend/counter
            print(y,x,base[y,x],"->",f[y,x])
    raise SystemExit
A.run_program('g50t', prog)
