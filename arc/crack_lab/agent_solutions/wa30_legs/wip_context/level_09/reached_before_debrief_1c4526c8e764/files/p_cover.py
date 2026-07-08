import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
GOALS=json.load(open('/tmp/l8_goals.json'))
GOALS=[tuple(g) for g in GOALS]
def cellcolor(f,R,C):
    blk=f[R*4:R*4+4,C*4:C*4+4]; return set(int(v) for v in np.unique(blk))
def covered(f):
    # goal tile has a box if it contains a 9-core with 3 or 4 border (locked or placed)
    c=0; locked=0
    for (R,C) in GOALS:
        u=cellcolor(f,R,C)
        if 9 in u and (3 in u or 4 in u): c+=1
        if 3 in u: locked+=1
    return c,locked
env=l8(); base=env.levels_completed
for step in range(120):
    env.step(1 if step%2==0 else 2)
    if env.levels_completed>base: print("WIN",step);break
    if step%10==9:
        f=np.asarray(env.frame()); cov,lk=covered(f)
        print(step,"covered",cov,"/",len(GOALS),"locked3",lk)
# inspect a specific left goal tile
f=np.asarray(env.frame())
for g in [(12,4),(2,3),(13,4)]:
    print("tile",g,"colors",sorted(cellcolor(f,*g)))
