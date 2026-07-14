import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    c=env.clone()
    for a in [4,4,4,4]: c.step(a)  # to (8,38)
    c.step(5)   # USE: stamp 2 at (8,38)? avatar reset
    f=np.asarray(c.frame())
    print("after USE at (8,38): avatar/2 bodies")
    r9=[b.bbox for b in P.connected_components(f,colors=[9]) if 7<=b.bbox[0]<49 and b.bbox[3]!=63 and b.area>=15]
    r2=[b.bbox for b in P.connected_components(f,colors=[2])]
    print("9:",r9,"2:",r2)
    print("=== maze rows 7-44 ===")
    for r in range(7,45): print(f"{r:2d} "+''.join(str(int(v)) for v in f[r,13:52]))
    raise SystemExit
A.run_program('g50t', prog)
