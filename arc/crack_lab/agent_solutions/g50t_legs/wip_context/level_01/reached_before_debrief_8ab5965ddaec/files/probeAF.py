import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    c=env.clone()
    for a in [4,4,4,4]: c.step(a)  # (8,38) eat config
    f=np.asarray(c.frame())
    print("=== (8,38) eat config maze ===")
    for r in range(7,58):
        print(f"{r:2d} "+''.join(str(int(v)) for v in f[r,13:52]))
    return
A.run_program('g50t', prog)
