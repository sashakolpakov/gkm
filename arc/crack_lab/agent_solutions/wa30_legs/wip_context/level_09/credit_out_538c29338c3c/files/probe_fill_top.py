import numpy as np, legs
from l8env import l8
env=l8()
for a in [4,4,4,1,1,1,1]:  # into top arena
    env.step(a)
targets=[(2,12),(3,12),(2,13),(3,13),(2,11),(3,11),(2,14),(3,14)]
for drop in targets:
    av,boxes,walls=legs._grid_scan(env)
    free=[b for b in boxes if b not in targets and b[0]<7]
    if not free: 
        print('no free top boxes'); break
    free.sort(key=lambda b: abs(b[0]-drop[0])+abs(b[1]-drop[1]))
    before=len(boxes)
    ok=legs.carry_box_to(env,free[0],drop,cap=40)
    av,boxes,walls=legs._grid_scan(env)
    print('drop',drop,'ok',ok,'nboxes',len(boxes),'lvl',env.levels_completed,'steps',len(env.path)-466,'term',env.terminal())
    if env.terminal() or env.levels_completed>7: break
