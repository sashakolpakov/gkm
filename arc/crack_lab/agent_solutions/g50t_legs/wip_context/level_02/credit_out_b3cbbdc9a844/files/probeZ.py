import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    base=np.asarray(env.frame())
    for a in [0,6,7,8,9]:
        c=env.clone()
        try:
            c.step(a)
            d=P.frame_delta(base,c.frame())
            print("step(",a,") delta",d['count'],d['bbox'],"lvl",c.levels_completed,"term",c.terminal())
        except Exception as e:
            print("step(",a,") ERR",repr(e)[:80])
    # coordinate clicks on interesting cells
    for (x,y) in [(38,16),(41,20),(46,52),(52,46),(2,2),(2,6)]:
        c=env.clone()
        try:
            c.step(6,x,y)
            d=P.frame_delta(base,c.frame())
            print("step(6,",x,y,") delta",d['count'],"lvl",c.levels_completed)
        except Exception as e:
            print("step(6,",x,y,") ERR",repr(e)[:60])
    raise SystemExit
A.run_program('g50t', prog)
