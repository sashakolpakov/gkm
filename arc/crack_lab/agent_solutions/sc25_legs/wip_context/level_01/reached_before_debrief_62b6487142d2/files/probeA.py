import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def main(env):
    f = np.asarray(env.frame())
    print("   "+''.join(str(c%10) for c in range(8,45)))
    for r in range(46,66):
        print(f"{r:2d} "+''.join('.' if f[r,c]==0 else format(f[r,c],'x') for c in range(8,45)))
A.run_program('sc25', main)
