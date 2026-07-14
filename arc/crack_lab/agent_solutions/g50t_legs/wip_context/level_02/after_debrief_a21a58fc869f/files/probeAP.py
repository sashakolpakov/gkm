import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def blobs(f):
    b9=[b.bbox for b in P.connected_components(f,colors=[9]) if 7<=b.bbox[0]<49 and b.bbox[3]!=63 and b.area>=8]
    b2=[b.bbox for b in P.connected_components(f,colors=[2]) if b.bbox[0]>=7]
    return b9,b2
def prog(env):
    c=env.clone()
    c.step(2); c.step(5); c.step(4)  # phase B, spawn 2 at (14,14), 9 at (8,20)
    print("base", blobs(np.asarray(c.frame())))
    for a in (1,2,3,4):
        cc=c.clone(); cc.step(a)
        print(P.ACTION_NAME[a], blobs(np.asarray(cc.frame())))
    # Also: does 9 moving onto the 2 marker interact? move 9 to (14,14) area
    raise SystemExit
A.run_program('g50t', prog)
