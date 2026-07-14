import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def two(f):
    return [b.bbox for b in P.connected_components(f,colors=[2]) if b.bbox[0]>=7]
def nine(f):
    return [b.bbox for b in P.connected_components(f,colors=[9]) if 7<=b.bbox[0]<49 and b.bbox[3]!=63 and b.area>=15]
def prog(env):
    tests={
      "(8,20)":[4],
      "(8,32)":[4,4,4],
      "(8,38)":[4,4,4,4],
      "(20,14)":[2,2],
      "(26,14)":[2,2,2],
      "(20,20)":[2,2,4],
      "(14,26)":[4,4,2],
    }
    for name,path in tests.items():
        c=env.clone()
        for a in path: c.step(a)
        c.step(5)  # USE
        # try each first move that is valid; use DOWN then if no move try RIGHT
        for firstmove in [2,4,1,3]:
            cc=c.clone(); cc.step(firstmove)
            f=np.asarray(cc.frame())
            if two(f):
                print(f"9split={name} firstmove={P.ACTION_NAME[firstmove]} -> 2 at {two(f)} 9 at {nine(f)}")
                break
    raise SystemExit
A.run_program('g50t', prog)
