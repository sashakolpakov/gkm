import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
def main(env):
    f=np.asarray(env.frame())
    print("ff-box rows50-59 cols11-20:")
    print("    "+''.join(str(c%10) for c in range(11,21)))
    for r in range(50,60):
        print(f" {r:2d} "+''.join('.' if f[r,c]==0 else format(f[r,c],'x') for c in range(11,21)))
A.run_program('sc25', main)
