import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    c=env.clone()
    c.step(4); c.step(3)  # back to (8,14), counter+1
    f1=np.asarray(c.frame()).copy()
    c.step(4); c.step(3)  # again
    f2=np.asarray(c.frame()).copy()
    d=P.frame_delta(f1,f2)
    print("counter-only delta count",d['count'],"bbox",d['bbox'])
    print("samples",d['samples'])
    raise SystemExit
A.run_program('g50t', prog)
