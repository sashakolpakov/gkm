import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def main(env):
    base=np.asarray(env.frame()).copy()
    hits=[]
    for x in range(64):
        for y in range(64):
            c=env.clone(); c.step(6,x,y)
            f=np.asarray(c.frame())
            if c.levels_completed>0 or not np.array_equal(f,base):
                hits.append((x,y,int((f!=base).sum()),c.levels_completed))
    print("total hits",len(hits))
    for h in hits[:20]: print(h)
A.run_program('sc25', main)
