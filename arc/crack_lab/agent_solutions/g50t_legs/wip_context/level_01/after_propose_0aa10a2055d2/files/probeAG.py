import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    # enter phase B: need to USE, but USE at start=0 (no effect). USE only works after moving.
    c=env.clone()
    c.step(2)   # down to (14,14)
    c.step(5)   # USE -> phase B, avatar home (8,14)
    print("phase B legend rows0-6:")
    f=np.asarray(c.frame())
    for r in range(6): print(''.join(str(int(v)) for v in f[r,0:10]))
    base=f.copy()
    for a in (1,2,3,4):
        cc=c.clone(); cc.step(a)
        d=P.frame_delta(base,cc.frame())
        # report changes in legend region vs maze region
        leg=int((base[0:6,0:12]!=np.asarray(cc.frame())[0:6,0:12]).sum())
        print("phaseB move",P.ACTION_NAME[a],"total",d['count'],"legendchg",leg,"bbox",d['bbox'])
    raise SystemExit
A.run_program('g50t', prog)
