import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    c=env.clone()
    for i in range(140):
        a=4 if i%2==0 else 3
        c.step(a)
        if c.terminal():
            print("clone terminal at step",i,"c1",P.color_counts(c.frame()).get(1)); break
    else:
        print("no terminal in 140, c1",P.color_counts(c.frame()).get(1),"c9",P.color_counts(c.frame()).get(9))
    raise SystemExit
A.run_program('g50t', prog)
