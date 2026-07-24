import sys, numpy as np
sys.path.insert(0,'/Users/sasha/gkm/arc/crack_lab')
import gkm_arena as A
import perception as P
def prog(env):
    # tour: cover all reachable positions & pushes & USE, on REAL env
    tour=[]
    tour+= [4,4,4,4]          # to (8,38) eat/push
    tour+= [3,3,3,3]          # back to (8,14)
    tour+= [2,2,2,2]          # down to (32,14)
    tour+= [1,1,1,1]          # up
    tour+= [2,4]              # (14,14)->(14,26)? 
    tour+= [2,4,2]            # explore middle
    tour+= [5]                # USE (phase B)
    tour+= [4,2,3,1]          # moves in phase B
    tour+= [5]                # USE back
    tour+= [2,2,2,4,3,1]
    last=-1
    for i,a in enumerate(tour):
        env.step(a)
        if env.levels_completed!=last:
            print("step",i,"act",a,"lvl",env.levels_completed)
            last=env.levels_completed
        if env.levels_completed>0:
            print("!!! WIN"); break
        if env.terminal():
            print("terminal at",i); break
    print("final lvl",env.levels_completed,"term",env.terminal())
    prog.d=1
A.run_program('g50t', prog)
