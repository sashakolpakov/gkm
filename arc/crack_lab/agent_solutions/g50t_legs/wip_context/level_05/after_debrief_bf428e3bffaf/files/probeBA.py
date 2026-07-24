import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
# Reconsider: maybe step returns a reward. Inspect return value of env.step.
def prog(env):
    for a in [2,4,4,4,5,3,2]:
        r=env.step(a)
        # r appears to be the frame (ndarray). check type
        print("act",a,"ret type",type(r).__name__, "shape" , getattr(r,'shape',None), "lvl",env.levels_completed)
    prog.d=1
A.run_program('g50t', prog)
