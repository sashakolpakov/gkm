import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def av(e):
    for b in P.connected_components(e.frame(),colors=[9]):
        if 7<=b.bbox[0]<49 and b.bbox[3]!=63 and b.area>=18: return b.bbox[:2]
    return None
def make():
    def prog(env):
        seq=[4,4,4,4, 3,3,3,3, 2,2,2,2, 5, 4,4,4,4, 5, 2,2,2,2,5,
             4,2,4,2,3,1,3,1,5,4,4,2,3,1,5,2,4,3]
        log=[]
        for a in seq:
            env.step(a)
            if env.levels_completed>0:
                log.append(("WIN",a,av(env),env.levels_completed)); break
            if env.terminal():
                log.append(("TERM",a,env.levels_completed)); break
        prog.log=log
        prog.final=(env.levels_completed, env.terminal())
    return prog
p=make(); A.run_program('g50t', p)
print("final",p.final)
print("events",p.log)
