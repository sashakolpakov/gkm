import sys, numpy as np, json
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
paths=json.load(open("paths25.json"))
def make(path):
    def prog(env):
        for a in path: 
            env.step(a)
            if env.levels_completed>0: break
        prog.res=(env.levels_completed, env.terminal())
    return prog
for i,path in enumerate(paths):
    p=make(path); A.run_program('g50t', p)
    if p.res[0]>0 or True:
        pass
    print(i, "len",len(path), "lvl",p.res[0], "term",p.res[1], path if p.res[0]>0 else "")
