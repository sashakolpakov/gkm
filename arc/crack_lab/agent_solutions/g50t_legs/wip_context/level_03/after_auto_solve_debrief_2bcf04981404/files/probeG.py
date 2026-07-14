import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P

def prog(env):
    print("env.actions",env.actions)
    for xy in [(30,30),(46,52),(10,16)]:
        c=env.clone()
        try:
            r=c.step(6,xy[0],xy[1])
            f=np.asarray(c.frame()); base=np.asarray(env.frame())
            d=P.frame_delta(base,f)
            print("step6",xy,"ret",r,"delta",d['count'],d['bbox'])
        except Exception as e:
            print("step6",xy,"ERR",repr(e))
    raise SystemExit
A.run_program('g50t', prog)
