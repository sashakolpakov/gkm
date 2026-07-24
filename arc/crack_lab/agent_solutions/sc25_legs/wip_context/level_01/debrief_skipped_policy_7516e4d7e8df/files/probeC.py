import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def marker(f):
    return '/'.join(''.join(format(f[r,c],'x') for c in range(39,43)) for r in range(19,23))

def main(env):
    for a in [1,2,3,4]:
        for reps in [3]:
            c = env.clone()
            for i in range(reps):
                c.step(a)
            f=np.asarray(c.frame())
            print(f"action {a} x{reps}: marker={marker(f)}")
A.run_program('sc25', main)
