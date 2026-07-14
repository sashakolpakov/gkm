import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    f=np.asarray(env.frame())
    print("start hash",hash(f.tobytes())%100000,"c8",P.color_counts(f).get(8))
    raise SystemExit
for _ in range(3):
    A.run_program('g50t', prog)
