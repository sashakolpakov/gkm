import sys,json; sys.path.insert(0,'.')
from mkstate import l8
import numpy as np
path=json.load(open('/tmp/l8_minagent.json'))
env=l8()
for a in path: env.step(a)
base=env.levels_completed
def fill(f):
    return int((f[8:16,44:60]==4).sum()),int((f[48:60,48:60]==4).sum())
# yield: avatar move harmlessly (it's at bottom region ~ (11,?)). Use small wiggle.
for i in range(90):
    if env.terminal(): print("terminal at",i); break
    env.step(1 if i%2==0 else 2)
    if env.levels_completed>base: print("WIN at idle",i); break
    if i%10==9:
        f=np.asarray(env.frame()); print(i,"fill",fill(f),"lvl",env.levels_completed)
print("final",env.levels_completed)
