import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    base=np.asarray(env.frame())
    hits=[]
    for x in range(0,64,1):
        for y in range(0,64,1):
            c=env.clone()
            try:
                c.step(6,x,y)
            except Exception as e:
                print("err",e); raise SystemExit
            f=np.asarray(c.frame())
            n=int((base!=f).sum())
            if n>0:
                hits.append((x,y,n,c.levels_completed,c.terminal()))
    print("num hit cells",len(hits))
    for h in hits[:60]: print(h)
    raise SystemExit
A.run_program('g50t', prog)
