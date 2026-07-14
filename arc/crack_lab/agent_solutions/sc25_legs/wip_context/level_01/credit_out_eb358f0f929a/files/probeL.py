import sys, random
sys.path.insert(0, '/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import numpy as np

def main(env):
    best=0
    for trial in range(400):
        c=env.clone()
        for i in range(70):
            c.step(random.choice([1,2,3,4]))
            if c.levels_completed>0:
                print("WIN trial",trial,"step",i); best=1; break
            if c.terminal(): break
        if best: break
    if not best:
        print("no win in random rollouts")
        # also test moving avatar far RIGHT
        c=env.clone()
        for i in range(20): c.step(4)
        f=np.asarray(c.frame())
        cols=[cc for cc in range(17,64) if f[20,cc] in (9,10)]
        print("rightmost avatar cols after 20 right:",(min(cols),max(cols)) if cols else None,"lvl",c.levels_completed)
A.run_program('sc25', main)
