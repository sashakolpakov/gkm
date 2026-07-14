import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def main(env):
    base=np.asarray(env.frame()).copy()
    # settle baseline: what does a lone settle produce? use step(1) then step(1)
    hits=[]
    for x in range(64):
        for y in range(64):
            c=env.clone()
            c.step(6,x,y)
            c.step(6,x,y)  # settle with same 6 (if 6 no-op, reveals prior 6? but 6 is what we test)
            f=np.asarray(c.frame()).copy(); f[:,60:]=base[:,60:]
            if c.levels_completed>0 or not np.array_equal(f,base):
                hits.append((x,y,int((f!=base).sum()),c.levels_completed))
    print("hits",len(hits))
    for h in hits[:20]: print(h)
A.run_program('sc25', main)
