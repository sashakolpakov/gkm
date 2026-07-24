import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def show(f,r0,r1,c0,c1):
    print("   "+''.join(str(c%10) for c in range(c0,c1)))
    for r in range(r0,r1):
        print(f"{r:2d} "+''.join('.' if f[r,c]==0 else format(f[r,c],'x') for c in range(c0,c1)))

def main(env):
    c=env.clone()
    for i in range(15): c.step(1) # many up
    print("UP15 lvl",c.levels_completed)
    show(np.asarray(c.frame()),10,24,35,46)
    c=env.clone()
    for i in range(15): c.step(2) # many down
    print("DOWN15 lvl",c.levels_completed)
    show(np.asarray(c.frame()),18,34,35,46)
A.run_program('sc25', main)
