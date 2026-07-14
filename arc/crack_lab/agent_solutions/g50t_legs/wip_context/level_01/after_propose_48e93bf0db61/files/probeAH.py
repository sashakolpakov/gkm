import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    c=env.clone()
    c.step(2); c.step(5)   # phase B, avatar home (8,14)
    print("=== phase B start maze ===")
    f=np.asarray(c.frame())
    for r in range(7,45): print(f"{r:2d} "+''.join(str(int(v)) for v in f[r,13:52]))
    base=f.copy()
    c.step(4)  # RIGHT in phase B
    f2=np.asarray(c.frame())
    print("=== after RIGHT(phaseB) changes ===")
    ys,xs=np.where(base!=f2)
    for y,x in zip(ys,xs):
        if y<7 or y==63: continue
        print(y,x,base[y,x],"->",f2[y,x])
    raise SystemExit
A.run_program('g50t', prog)
