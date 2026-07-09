import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np, legs
path=json.load(open('/tmp/l8_minagent.json'))
env=l8()
for a in path[:20]: env.step(a)
n_after_prefix=len(env.path)-466
base=env.levels_completed
# free penned box (2,1): carry to open top region so courier can seat
def boxes_topleft(f):
    out=[]
    for R in range(0,5):
        for C in range(0,4):
            blk=f[R*4:R*4+4,C*4:C*4+4];u=set(int(v) for v in np.unique(blk))
            if 9 in u and 4 in u: out.append((R,C))
    return out
f=np.asarray(env.frame()); print("penned left:",boxes_topleft(f),"moves so far",len(env.path)-466)
# deliver each penned box into top container directly (avatar), courier locks
for b in boxes_topleft(f):
    ok=legs.carry_box_to(env,b,(3,12),cap=50)
    print("deliver",b,"ok",ok,"moves",len(env.path)-466,"lvl",env.levels_completed)
    if env.levels_completed>base: break
# yield in corner
for i in range(80):
    if env.terminal() or env.levels_completed>base: break
    env.step(3 if i%2==0 else 4)
print("FINAL lvl",env.levels_completed,"total level8 moves",len(env.path)-466)
