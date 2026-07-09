import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
path=json.load(open('/tmp/l8_minagent.json'))
env=l8()
for a in path[:20]: env.step(a)
for i in range(100): 
    if env.terminal(): break
    env.step(3 if i%2==0 else 4)  # wiggle in place (avatar at (11,7)->left/right)
f=np.asarray(env.frame())
print("lvl",env.levels_completed)
def dump(r0,r1,c0,c1,lbl):
    print(lbl)
    for r in f[r0:r1,c0:c1]: print(" ".join(f"{int(v):2d}" for v in r))
dump(8,20,4,16,"TOP-LEFT pen")
dump(48,60,12,24,"BOT-LEFT 3x3")
# color counts
for c in [2,3,4,7,9,12]: print("c",c,int((f==c).sum()))
