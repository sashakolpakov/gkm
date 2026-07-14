import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def blobs(f):
    b9=[b.bbox[:2] for b in P.connected_components(f,colors=[9]) if 7<=b.bbox[0]<49 and b.bbox[3]!=63 and b.area>=8]
    b2=[b.bbox[:2] for b in P.connected_components(f,colors=[2]) if b.bbox[0]>=7]
    return b9,b2
def prog(env):
    # spawn 2 via DOWN then probe every subsequent single move for dual motion
    c=env.clone(); c.step(2); c.step(5); c.step(2)  # phaseB, first move DOWN
    print("after spawn:", blobs(np.asarray(c.frame())))
    for a in (1,2,3,4):
        cc=c.clone(); cc.step(a)
        print(" press",P.ACTION_NAME[a],"->",blobs(np.asarray(cc.frame())))
    # try a longer varied sequence, print both each step
    print("--- long seq ---")
    c2=env.clone(); c2.step(2); c2.step(5); c2.step(2)
    for a in [4,4,2,1,3]:
        c2.step(a); print(P.ACTION_NAME[a], blobs(np.asarray(c2.frame())))
    raise SystemExit
A.run_program('g50t', prog)
