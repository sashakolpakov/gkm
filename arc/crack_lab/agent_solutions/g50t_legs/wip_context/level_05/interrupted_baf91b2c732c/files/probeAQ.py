import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def run(env, reach, burn):
    c=env.clone()
    for a in reach: c.step(a)
    steps=0
    while not c.terminal() and steps<400:
        c.step(burn); steps+=1
    return c.levels_completed, c.terminal(), steps
def prog(env):
    tests={
      "(32,14)wallDOWN":([2,2,2,2],2),
      "(8,38)eat_wallRIGHT":([4,4,4,4],4),
      "(8,14)wallUP":([],1),
      "(26,14)wallLEFT":([2,2,2],3),
      "(20,26)wallUP":([4,4,2,2],1),
    }
    for name,(reach,burn) in tests.items():
        r=run(env,reach,burn)
        print(name,"-> lvl",r[0],"term",r[1],"burnsteps",r[2])
    raise SystemExit
A.run_program('g50t', prog)
