import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def main(env):
    base=np.asarray(env.frame()).copy()
    for a in [1,2,3,4]:
        c=env.clone(); c.step(a); c.step(a)
        f=np.asarray(c.frame()).copy()
        f[:,60:]=base[:,60:] # ignore timer
        ys,xs=np.where(f!=base)
        regions=sorted(set((int(y),int(x),int(base[y,x]),int(f[y,x])) for y,x in zip(ys,xs)))
        print(f"action {a}: {len(regions)} changed cells")
        for rgn in regions[:12]: print("   ",rgn)
A.run_program('sc25', main)
