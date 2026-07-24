import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def show(f):
    print("   "+''.join(str(c%10) for c in range(8,45)))
    for r in range(16,24):
        print(f"{r:2d} "+''.join('.' if f[r,c]==0 else format(f[r,c],'x') for c in range(8,45)))

def main(env):
    c=env.clone()
    for i in range(12):
        c.step(3)  # left repeatedly
    print("level", c.levels_completed, "term", c.terminal())
    show(np.asarray(c.frame()))
A.run_program('sc25', main)
