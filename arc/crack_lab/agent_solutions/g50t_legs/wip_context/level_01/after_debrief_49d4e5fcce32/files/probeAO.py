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
    c.step(2); c.step(5)  # phase B
    print("phaseB start", blobs(np.asarray(c.frame())))
    for a in [4,4,4,4,2,2,2]:
        c.step(a)
        b9,b2=blobs(np.asarray(c.frame()))
        print(P.ACTION_NAME[a],"9:",b9,"2:",b2)
    raise SystemExit
A.run_program('g50t', prog)
