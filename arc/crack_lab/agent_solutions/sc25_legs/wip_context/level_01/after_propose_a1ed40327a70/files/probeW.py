import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
def reservoir(f): return int((f[:,62:64]==14).sum())
def show_ck(f):
    for r in range(47,64):
        print(f" {r:2d} "+''.join('.' if f[r,c]==0 else format(f[r,c],'x') for c in range(22,39)))
def run(env, clicks, dummy=True):
    c=env.clone()
    if dummy: c.step(1)      # absorb first action
    for (x,y) in clicks: c.step(6,x,y)
    c.step(1)                # reveal
    return c
def main(env):
    N=(30,50); W=(25,55); E=(35,55); S=(30,60)
    c=run(env,[N,W,E,S])
    f=np.asarray(c.frame())
    print("fill 4 edges: lvl",c.levels_completed,"res",reservoir(f),"term",c.terminal())
    show_ck(f)
A.run_program('sc25', main)
