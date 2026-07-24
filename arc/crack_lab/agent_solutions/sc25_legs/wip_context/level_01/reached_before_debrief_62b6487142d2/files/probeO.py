import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def main(env):
    base=np.asarray(env.frame()).copy()
    for n in range(0,8):
        c=env.clone()
        for i in range(n): c.step(3)
        c.step(2) # settle with a down (orientation flip, harmless) to reveal
        f=np.asarray(c.frame())
        # diff bottom region rows 45-64 cols 0-45
        sub_b=base[45:64,0:45]; sub_f=f[45:64,0:45]
        d=int((sub_b!=sub_f).sum())
        print(f"lefts={n} bottom_diff={d} lvl={c.levels_completed}")
A.run_program('sc25', main)
