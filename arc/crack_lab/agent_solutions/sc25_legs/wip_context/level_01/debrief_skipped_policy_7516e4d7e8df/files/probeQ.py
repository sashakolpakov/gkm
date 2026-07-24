import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np
def timer(f): return int((f[:,62]==14).sum())
def main(env):
    # does action 6 consume turns?
    c=env.clone()
    for i in range(10): c.step(6,20,20)
    c.step(1) # settle
    print("after 10x step6 + settle: timer",timer(np.asarray(c.frame())),"lvl",c.levels_completed,"term",c.terminal())
    # baseline: 1 move settle
    c=env.clone(); c.step(1); c.step(1)
    print("after 2x up: timer",timer(np.asarray(c.frame())))
    # 6 with 2-arg?
    for args in [(6,20),(6,)]:
        try:
            c=env.clone(); c.step(*args); c.step(1)
            print("step",args,"ok timer",timer(np.asarray(c.frame())))
        except Exception as e:
            print("step",args,"ERR",repr(e)[:60])
A.run_program('sc25', main)
