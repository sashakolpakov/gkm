import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def show(f):
    print("   "+''.join(str(c%10) for c in range(10,45)))
    for r in range(16,24):
        print(f"{r:2d} "+''.join('.' if f[r,c]==0 else format(f[r,c],'x') for c in range(10,45)))

def main(env):
    print("=== base ===")
    show(np.asarray(env.frame()))
    for a,name in [(3,'left3'),(4,'right3')]:
        c=env.clone()
        for i in range(3): c.step(a)
        print(f"=== {name} ===")
        show(np.asarray(c.frame()))
A.run_program('sc25', main)
