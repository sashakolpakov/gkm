import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
import numpy as np

def main(env):
    base = np.asarray(env.frame()).copy()
    # test return value of step
    clone = env.clone()
    r = clone.step(6,13,19)
    print("step return:", r)
    r = clone.step(1)
    print("step1 return:", r)
    # scan grid coarsely for changes with action 6
    hits=[]
    for x in range(0,64,2):
        for y in range(0,64,2):
            c = env.clone()
            c.step(6,x,y)
            d = P.frame_delta(base, c.frame())
            if d['count']>0:
                hits.append((x,y,d['count']))
    print("hits count:", len(hits))
    for h in hits[:30]:
        print(h)

A.run_program('sc25', main)
