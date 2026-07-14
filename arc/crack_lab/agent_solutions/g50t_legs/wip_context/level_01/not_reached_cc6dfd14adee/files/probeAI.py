import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def bodies(f):
    r9=[b.bbox for b in P.connected_components(f,colors=[9]) if 7<=b.bbox[0]<49 and b.bbox[3]!=63 and b.area>=15]
    r2=[b.bbox for b in P.connected_components(f,colors=[2]) if b.area>=8]
    return r9,r2
def prog(env):
    c=env.clone()
    c.step(2); c.step(5)   # phase B
    for i,a in enumerate([4,4,2,2,3,1,5]):
        c.step(a)
        f=np.asarray(c.frame())
        r9,r2=bodies(f)
        print(i,"act",P.ACTION_NAME[a],"9:",r9,"2:",r2,"lvl",c.levels_completed,"legrow1",''.join(str(int(v)) for v in f[1,1:8]))
    raise SystemExit
A.run_program('g50t', prog)
