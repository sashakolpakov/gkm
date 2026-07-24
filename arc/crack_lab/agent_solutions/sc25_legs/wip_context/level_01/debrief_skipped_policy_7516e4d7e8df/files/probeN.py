import sys
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def timer(f): return int((f[:,62]==14).sum())
def main(env):
    c=env.clone()
    for i in range(200):
        c.step(1)
        if c.terminal():
            print("terminal at step",i,"lvl",c.levels_completed); sys.stdout.flush(); return
        if i%20==0:
            print("step",i,"timer",timer(np.asarray(c.frame())),"lvl",c.levels_completed); sys.stdout.flush()
    print("done no terminal, lvl",c.levels_completed)
A.run_program('sc25', main)
