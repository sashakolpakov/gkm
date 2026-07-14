import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def make():
    def prog(env):
        f=np.asarray(env.frame())
        prog.h=(int(f.sum()), P.color_counts(f).get(8))
    return prog
for i in range(3):
    p=make()
    A.run_program('g50t', p)
    print(i, p.h)
