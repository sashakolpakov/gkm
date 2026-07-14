import sys
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
env=A.Arena('g50t')
b=env.frame().copy()
try:
    env.step(6,20,10)
    d=P.frame_delta(b,env.frame())
    print("a6(20,10) diff",d['count'],d['bbox'],d['samples'][:8],"lvl",env.levels_completed)
except Exception as e:
    print("err",e)
