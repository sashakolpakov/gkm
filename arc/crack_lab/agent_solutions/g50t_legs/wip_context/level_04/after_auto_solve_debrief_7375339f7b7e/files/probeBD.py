import sys, numpy as np, itertools, random
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
# Try many distinct ROOT-env playthroughs (each a fresh run_program) with
# structured strategies; report if levels_completed EVER > 0.
strategies=[]
# systematic: for each reachable base config path, extend with all single actions x2
base_paths=[[],[4],[4,4],[4,4,4],[4,4,4,4],[2],[2,2],[2,2,2],[2,2,2,2],
            [2,5],[2,5,4],[2,5,2],[4,4,4,4,5],[4,4,4,4,3],[2,2,2,2,5]]
import json
extra=[list(p) for p in itertools.product([1,2,3,4,5],repeat=3)]
random.seed(0); random.shuffle(extra); extra=extra[:40]
for bp in base_paths:
    for e in extra:
        strategies.append(bp+e)
best=0
def make(path):
    def prog(env):
        for a in path:
            env.step(a)
            if env.levels_completed>0: 
                prog.win=True; return
            if env.terminal(): break
        prog.win=(env.levels_completed>0)
        prog.lvl=env.levels_completed
    return prog
wins=0
for i,path in enumerate(strategies):
    p=make(path); A.run_program('g50t', p)
    if getattr(p,'lvl',0)>0:
        wins+=1; print("WIN", path)
print("tested",len(strategies),"root playthroughs; wins:",wins)
