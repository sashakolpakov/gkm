import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
def main(env):
    # baseline: two dummy ups
    b=env.clone(); b.step(1); b.step(1); base=np.asarray(b.frame()).copy()
    regions={}
    for x in range(64):
        for y in range(64):
            c=env.clone(); c.step(1); c.step(6,x,y); c.step(1)
            f=np.asarray(c.frame())
            diff=(f!=base)&(np.arange(64)[None,:]<60)
            if diff.any():
                ys,xs=np.where(diff)
                key=(int(ys.min())//5, int(xs.min())//5)
                regions.setdefault(key,[]).append((x,y,int(diff.sum())))
    for k in sorted(regions):
        v=regions[k]
        print("region",k,"n_coords",len(v),"sample",v[0])
A.run_program('sc25', main)
