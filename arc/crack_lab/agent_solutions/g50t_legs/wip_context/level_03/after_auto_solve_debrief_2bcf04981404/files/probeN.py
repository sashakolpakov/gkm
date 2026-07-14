import sys, numpy as np, random
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def av(env):
    for b in P.connected_components(env.frame(),colors=[9]):
        if b.area==24: return b.bbox[:2]
    return None
def prog(env):
    random.seed(1)
    positions=set()
    for i in range(150):
        a=random.choice([1,2,3,4,5])
        env.step(a)
        positions.add(av(env))
        if env.levels_completed>0 or env.terminal():
            print("EVENT at step",i,"act",a,"lvl",env.levels_completed,"term",env.terminal())
            break
    cc=P.color_counts(env.frame())
    print("final lvl",env.levels_completed,"term",env.terminal(),"counts",cc)
    print("distinct av positions visited",len(positions),sorted(x for x in positions if x))
    raise SystemExit
A.run_program('g50t', prog)
