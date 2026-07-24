import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def show_ck(f):
    print("     "+''.join(str(c%10) for c in range(22,39)))
    for r in range(47,64):
        print(f" {r:2d} "+''.join('.' if f[r,c]==0 else format(f[r,c],'x') for c in range(22,39)))

def main(env):
    base=np.asarray(env.frame()).copy()
    print("=== BASE checkerboard ===")
    show_ck(base)
    # click center (col=30,row=50) i.e. step(6,30,50)
    c=env.clone(); c.step(6,30,50); c.step(6,30,50)
    print("=== after click (col30,row50) ===")
    show_ck(np.asarray(c.frame()))
A.run_program('sc25', main)
