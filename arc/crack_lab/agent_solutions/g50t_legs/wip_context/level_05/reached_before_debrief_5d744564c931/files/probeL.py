import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
def prog(env):
    f=np.asarray(env.frame())
    print("cols header:", ''.join(str(c%10) for c in range(13,52)))
    for r in range(7,58):
        print(f"{r:2d} "+''.join(str(int(v)) for v in f[r,13:52]))
    raise SystemExit
A.run_program('g50t', prog)
