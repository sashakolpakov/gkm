import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def show(f):
    for row in np.asarray(f):
        print(''.join('.' if v==0 else format(v,'x') for v in row))

def main(env):
    base = np.asarray(env.frame()).copy()
    c = env.clone()
    for i in range(20):
        c.step(1)
    diff = int((np.asarray(c.frame())!=base).sum())
    print("20x up diff=", diff, "level", c.levels_completed)
    # try all four dirs 30 times each fresh
    for a in [1,2,3,4]:
        c = env.clone()
        for i in range(30):
            c.step(a)
        diff = int((np.asarray(c.frame())!=base).sum())
        print(f"30x {a} diff={diff}")
    # long random sequence
    import random
    c = env.clone()
    for i in range(100):
        c.step(random.choice([1,2,3,4]))
    diff = int((np.asarray(c.frame())!=base).sum())
    print("100 random diff", diff)
A.run_program('sc25', main)
