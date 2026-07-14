import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    c=env.clone()
    for a in [4,4,4,4]: c.step(a)  # 9 to (8,38)
    c.step(5)  # USE: record split=(8,38), reset 9->(8,14), phase B
    c.step(2)  # DOWN: spawns 2 at (8,38)?, moves 9 to (14,14)
    f=np.asarray(c.frame())
    r9=[b.bbox for b in P.connected_components(f,colors=[9]) if 7<=b.bbox[0]<49 and b.bbox[3]!=63 and b.area>=15]
    r2=[b.bbox for b in P.connected_components(f,colors=[2])]
    print("9:",r9,"2:",r2)
    print("col14 rows37-43:", ''.join(str(int(f[r,14])) for r in range(37,44)))
    print("=== rows 7-44 ===")
    for r in range(7,45): print(f"{r:2d} "+''.join(str(int(v)) for v in f[r,13:52]))
    raise SystemExit
A.run_program('g50t', prog)
