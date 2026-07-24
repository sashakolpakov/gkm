import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
import perception as P

def main(env):
    base = np.asarray(env.frame()).copy()
    # step once, twice, thrice
    for n in [1,2,3,4,5]:
        c = env.clone()
        for i in range(n):
            c.step(1)
        d = P.frame_delta(base, c.frame())
        print(f"n={n} up count={d['count']} bbox={d['bbox']} samples={d['samples'][:8]}")
A.run_program('sc25', main)
